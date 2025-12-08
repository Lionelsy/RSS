# 今日论文推荐 - 2025-12-08

共 153 篇论文

---

## 1. Manifold-Aware Point Cloud Completion via Geodesic-Attentive Hierarchical Feature Learning

**论文链接:** [http://arxiv.org/abs/2512.05710v1](http://arxiv.org/abs/2512.05710v1)

**作者:** Jianan Sun, Dongzhihan Wang, Mingyu Fan

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种流形感知的点云补全框架，通过引入测地距离近似器和流形感知特征提取器两个关键模块，在特征学习过程中明确融入非线性几何信息，提高了重建点云的语义一致性和结构保真度。

### 背景

现有点云补全方法虽然能实现合理的全局形状重建，但通常依赖欧几里得邻近性，忽略了点云的内在非线性几何结构，导致次优的几何一致性和语义模糊。

### 目的

开发一种能够捕获点云潜在流形拓扑结构的点云补全方法，提高重建结果的几何一致性和语义清晰度。

### 方法

提出流形感知的点云补全框架，包含两个关键模块：1)测地距离近似器(GDA)，用于估计点间测地距离；2)流形感知特征提取器(MAFE)，利用基于测地距离的k-NN分组和测地关系注意力机制引导特征提取过程。

### 主要发现

通过整合测地感知关系注意力，所提方法能够促进重建点云的语义一致性和结构保真度，在基准数据集上的实验表明该方法在重建质量上始终优于最先进的方法。

### 结论

流形感知的点云补全框架能够有效捕获点云的非线性几何结构，显著提高重建质量，解决了现有方法依赖欧几里得邻近性导致的几何一致性和语义模糊问题。

### 翻译

点云补全旨在从部分或稀疏的3D观测中恢复几何上一致的形状。尽管最近的方法已经实现了合理的全局形状重建，但它们通常依赖于欧几里得邻近性，而忽略了点云的内在非线性几何结构，导致次优的几何一致性和语义模糊性。在本文中，我们提出了一个流形感知的点云补全框架，在特征学习管道中明确融入了非线性几何信息。我们的方法引入了两个关键模块：测地距离近似器(GDA)，用于估计点之间的测地距离以捕获潜在流形拓扑；以及流形感知特征提取器(MAFE)，利用基于测地距离的k-NN分组和测地关系注意力机制来引导分层特征提取过程。通过整合测地感知关系注意力，我们的方法在重建的点云中促进了语义一致性和结构保真度。在基准数据集上的大量实验表明，我们的方法在重建质量上始终优于最先进的方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决点云补全问题，即从不完整或稀疏的3D观测中恢复几何一致的形状。这个问题在现实中很重要，因为真实世界的3D扫描常因遮挡、传感器限制或环境干扰而变得不完整，这对自动驾驶、机器人、增强现实和文化遗产保护等应用中的高级感知和场景理解构成挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到现有方法依赖欧几里得邻近性而忽略了点云的非线性几何结构，导致次优的几何一致性。他们借鉴了DGCNN的k-NN思想、AdaPoinTr的几何块、PointCFormer的高维特征空间局部亲和力估计以及PointAttN的全局注意力机制，但改进为基于测地距离的邻域定义和特征提取，从而更好地捕获点云的内在流形结构。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是明确将非线性几何信息整合到特征学习管道中，使用基于锚点的测地距离来捕获潜在流形拓扑。整体流程包括：1) Geodesic Distance Approximator (GDA)构建局部邻近图并计算测地距离；2) Manifold-Aware Feature Extractor (MAFE)通过Geodesic Neighborhood Grouper、Geodesic-Relational Attention Transformer和Manifold Positional Embedding提取流形感知特征；3) Coarse Completion and Upsample Model进行粗略补全和细化；4) 使用Chamfer Distance作为损失函数进行多阶段监督。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 基于锚点的测地距离近似方法(GDA)高效捕获表面拓扑；2) 流形感知特征提取器(MAFE)整合测地指导到邻域分组、注意力和位置嵌入；3) 测地关系注意力Transformer(GRA-T)将测地距离融入局部注意力计算；4) 流形位置嵌入(MPE)编码全局几何上下文。相比之前工作，本文不再依赖欧几里得邻近性，而是使用测地距离定义邻域，并将流形信息整合到特征学习的多个阶段，显著提升了重建质量。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种流形感知的点云补全框架，通过基于测地距离的特征学习显著提升了重建点云的几何一致性和语义连贯性，在各种基准测试中超越了最先进的方法。'}


### 论文摘要

Point cloud completion seeks to recover geometrically consistent shapes from partial or sparse 3D observations. Although recent methods have achieved reasonable global shape reconstruction, they often rely on Euclidean proximity and overlook the intrinsic nonlinear geometric structure of point clouds, resulting in suboptimal geometric consistency and semantic ambiguity. In this paper, we present a manifold-aware point cloud completion framework that explicitly incorporates nonlinear geometry information throughout the feature learning pipeline. Our approach introduces two key modules: a Geodesic Distance Approximator (GDA), which estimates geodesic distances between points to capture the latent manifold topology, and a Manifold-Aware Feature Extractor (MAFE), which utilizes geodesic-based $k$-NN groupings and a geodesic-relational attention mechanism to guide the hierarchical feature extraction process. By integrating geodesic-aware relational attention, our method promotes semantic coherence and structural fidelity in the reconstructed point clouds. Extensive experiments on benchmark datasets demonstrate that our approach consistently outperforms state-of-the-art methods in reconstruction quality.

---

## 2. AQUA-Net: Adaptive Frequency Fusion and Illumination Aware Network for Underwater Image Enhancement

**论文链接:** [http://arxiv.org/abs/2512.05960v1](http://arxiv.org/abs/2512.05960v1)

**作者:** Munsif Ali, Najmul Hassan, Lucia Ventura, Davide Di Bari, Simonepietro Canese

**发布时间:** 2025-12-05

### GPT解析

### 总结

该论文提出了AQUA-Net模型，一种结合频率域和光照域双分支设计的水下图像增强方法，能有效处理水下图像的颜色失真、低对比度和雾化问题，同时保持较低计算复杂度，适合实时应用。

### 背景

水下图像常因波长相关的光吸收和散射而出现严重的颜色失真、低对比度和雾化问题；现有深度学习模型计算复杂度高，限制了其在实际水下实时应用中的部署。

### 目的

开发一种能有效增强水下图像质量且计算复杂度较低的模型，并构建高质量的真实世界水下视频数据集用于模型评估。

### 方法

提出AQUA-Net模型，包含残差编码器-解码器和两个辅助分支：1)频率融合编码器通过傅里叶域频率线索增强空间表示；2)光照感知解码器受Retinex理论启发，通过学习光照图分离反射率与光照效果；3)还构建了来自地中海的高分辨率水下视频数据集，包含具有真实视觉退化的深海条件。

### 主要发现

1)在多个基准数据集上，AQUA-Net在定性和定量评估方面与最先进技术相当，同时参数更少；2)消融研究证实频率和光照分支提供互补贡献，提高可见性和颜色表示；3)模型展示了强大的泛化能力和鲁棒性。

### 结论

AQUA-Net为实际水下成像应用提供了有效解决方案，能在保持较低计算复杂度的同时显著改善水下图像质量。

### 翻译

水下图像通常因波长相关的光吸收和散射而遭受严重的颜色失真、低对比度和雾化外观。同时，现有的深度学习模型表现出高的计算复杂度，这限制了它们在实际水下应用中的实时部署。为了解决这些挑战，本文提出了一种新颖的水下图像增强模型，称为自适应频率融合与光照感知网络（AQUA-Net）。它集成了一个残差编码器-解码器和两个辅助分支，分别在频率域和光照域工作。频率融合编码器通过来自傅里叶域的频率线索丰富了空间表示，并保留了精细纹理和结构细节。受Retinex理论启发，光照感知解码器通过学习到的光照图执行自适应曝光校正，该光照图将反射率与光照效果分离。这种空间、频率和光照的联合设计使模型能够在各种水下条件下恢复色彩平衡、视觉对比度和感知真实感。此外，我们提出了一个来自地中海的高分辨率、真实世界水下视频派生数据集，它捕捉了具有真实视觉退化的深海挑战条件，以实现深度学习模型的稳健评估和开发。在多个基准数据集上的大量实验表明，AQUA-Net在定性和定量评估方面与SOTA相当，同时使用的参数更少。消融研究进一步证实频率和光照分支提供了互补的贡献，提高了可见性和颜色表示。总体而言，所提出的模型展示了强大的泛化能力和鲁棒性，并为实际水下成像应用提供了有效的解决方案。


### 论文摘要

Underwater images often suffer from severe color distortion, low contrast, and a hazy appearance due to wavelength-dependent light absorption and scattering. Simultaneously, existing deep learning models exhibit high computational complexity, which limits their practical deployment for real-time underwater applications. To address these challenges, this paper presents a novel underwater image enhancement model, called Adaptive Frequency Fusion and Illumination Aware Network (AQUA-Net). It integrates a residual encoder decoder with dual auxiliary branches, which operate in the frequency and illumination domains. The frequency fusion encoder enriches spatial representations with frequency cues from the Fourier domain and preserves fine textures and structural details. Inspired by Retinex, the illumination-aware decoder performs adaptive exposure correction through a learned illumination map that separates reflectance from lighting effects. This joint spatial, frequency, and illumination design enables the model to restore color balance, visual contrast, and perceptual realism under diverse underwater conditions. Additionally, we present a high-resolution, real-world underwater video-derived dataset from the Mediterranean Sea, which captures challenging deep-sea conditions with realistic visual degradations to enable robust evaluation and development of deep learning models. Extensive experiments on multiple benchmark datasets show that AQUA-Net performs on par with SOTA in both qualitative and quantitative evaluations while using less number of parameters. Ablation studies further confirm that the frequency and illumination branches provide complementary contributions that improve visibility and color representation. Overall, the proposed model shows strong generalization capability and robustness, and it provides an effective solution for real-world underwater imaging applications.

---

## 3. 论文ID: 2512.05953v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.05953v1.json'

---

## 4. Underwater Image Reconstruction Using a Swin Transformer-Based Generator and PatchGAN Discriminator

**论文链接:** [http://arxiv.org/abs/2512.05866v1](http://arxiv.org/abs/2512.05866v1)

**作者:** Md. Mahbub Hasan Akash, Aria Tasnim Mridula, Sheekar Banerjee, Ishtiak Al Mamoon

**发布时间:** 2025-12-05

**备注:** This paper has been accepted for presentation at the IEEE 28th International Conference on Computer and Information Technology (ICCIT), December 2025

### GPT解析

### 总结

该研究提出了一种基于Swin Transformer和GAN的新型水下图像重建框架，有效解决了水下图像的颜色失真、低对比度和雾霾问题，并在定量和定性评估中展示了优于现有方法的结果。

### 背景

水下成像是海洋探索、环境监测和基础设施检查的重要工具，但水的波长相关吸收和散射会导致严重的图像退化，包括颜色失真、低对比度和雾霾效应。

### 目的

解决传统重建方法和基于卷积神经网络的方法在处理水下图像退化时的局限性，包括感受野有限和无法建模全局依赖性问题。

### 方法

提出了一种深度学习框架，将Swin Transformer架构集成到生成对抗网络(GAN)中，生成器采用包含Swin Transformer块的U-Net结构，使用PatchGAN判别器进行对抗训练，在EUVP数据集上进行训练和评估。

### 主要发现

定量结果显示PSNR为24.76 dB，SSIM为0.89，达到最先进性能；视觉结果有效恢复了颜色平衡、提高了对比度并减少了雾霾；消融研究证实Swin Transformer设计优于卷积替代方案。

### 结论

所提出的方法为各种海洋应用提供了强大的水下图像重建能力，具有实际应用价值。

### 翻译

水下成像对海洋探索、环境监测和基础设施检查至关重要。然而，水会导致严重的图像退化，通过波长相关的吸收和散射，造成颜色失真、低对比度和雾霾效应。传统重建方法和基于卷积神经网络的方法往往由于感受野有限和无法建模全局依赖性而无法充分应对这些挑战。本文提出了一种新的深度学习框架，将Swin Transformer架构集成到生成对抗网络(GAN)中用于水下图像重建。我们的生成器采用包含Swin Transformer块的U-Net结构，以捕获局部特征和整个图像中颜色校正所需的长距离依赖关系。PatchGAN判别器提供对抗训练以确保高频细节的保留。我们在包含不同质量配对水下图像的EUVP数据集上训练和评估了我们的模型。定量结果展示了最先进的性能，PSNR为24.76 dB，SSIM为0.89，代表了与现有方法相比的显著改进。视觉结果显示了有效的颜色平衡恢复、对比度提高和雾霾减少。消融研究证实了我们设计的Swin Transformer优于卷积替代方案。所提出的方法为各种海洋应用提供了强大的水下图像重建能力。


### 论文摘要

Underwater imaging is essential for marine exploration, environmental monitoring, and infrastructure inspection. However, water causes severe image degradation through wavelength-dependent absorption and scattering, resulting in color distortion, low contrast, and haze effects. Traditional reconstruction methods and convolutional neural network-based approaches often fail to adequately address these challenges due to limited receptive fields and inability to model global dependencies. This paper presented a novel deep learning framework that integrated a Swin Transformer architecture within a generative adversarial network (GAN) for underwater image reconstruction. Our generator employed a U-Net structure with Swin Transformer blocks to capture both local features and long-range dependencies crucial for color correction across entire images. A PatchGAN discriminator provided adversarial training to ensure high-frequency detail preservation. We trained and evaluated our model on the EUVP dataset, which contains paired underwater images of varying quality. Quantitative results demonstrate stateof-the-art performance with PSNR of 24.76 dB and SSIM of 0.89, representing significant improvements over existing methods. Visual results showed effective color balance restoration, contrast improvement, and haze reduction. An ablation study confirms the superiority of our Swin Transformer designed over convolutional alternatives. The proposed method offers robust underwater image reconstruction suitable for various marine applications.

---

## 5. Mechanistic Interpretability of Antibody Language Models Using SAEs

**论文链接:** [http://arxiv.org/abs/2512.05794v1](http://arxiv.org/abs/2512.05794v1)

**作者:** Rebonto Haque, Oliver M. Turnbull, Anisha Parsan, Nithin Parsan, John J. Yang, Charlotte M. Deane

**发布时间:** 2025-12-05

### GPT解析

### 总结

本研究探讨了稀疏自编码器(SAEs)在抗体语言模型p-IgGen中的应用，比较了TopK和有序SAEs两种方法在模型解释和生成控制方面的效果。

### 背景

稀疏自编码器(SAEs)是一种机械可解释性技术，已被用于揭示大型蛋白质语言模型中学习到的概念。

### 目的

采用TopK和有序SAEs来研究一个自回归抗体语言模型p-IgGen并引导其生成。

### 方法

使用TopK SAEs和有序SAEs两种技术来研究p-IgGen模型并尝试控制其生成。

### 主要发现

TopK SAEs能够揭示生物学上有意义的潜在特征，但高特征概念相关性并不保证对生成的因果控制；有序SAEs施加了层次结构，能够可靠地识别可引导的特征，但代价是更复杂且更难解释的激活模式。

### 结论

这些发现推进了领域特定蛋白质语言模型的机械可解释性，并表明对于将潜在特征映射到概念，TopK SAEs是足够的；当需要精确的生成引导时，有序SAEs更可取。

### 翻译

稀疏自编码器(SAEs)是一种机械可解释性技术，已被用于揭示大型蛋白质语言模型中学习到的概念。在此，我们采用TopK和有序SAEs来研究一个自回归抗体语言模型p-IgGen并引导其生成。我们表明，TopK SAEs可以揭示生物学上有意义的潜在特征，但高特征概念相关性并不保证对生成的因果控制。相比之下，有序SAEs施加了层次结构，能够可靠地识别可引导的特征，但代价是更复杂且更难解释的激活模式。这些发现推进了领域特定蛋白质语言模型的机械可解释性，并表明，虽然TopK SAEs足以将潜在特征映射到概念，但当需要精确的生成引导时，有序SAEs更可取。


### 论文摘要

Sparse autoencoders (SAEs) are a mechanistic interpretability technique that have been used to provide insight into learned concepts within large protein language models. Here, we employ TopK and Ordered SAEs to investigate an autoregressive antibody language model, p-IgGen, and steer its generation. We show that TopK SAEs can reveal biologically meaningful latent features, but high feature concept correlation does not guarantee causal control over generation. In contrast, Ordered SAEs impose an hierarchical structure that reliably identifies steerable features, but at the expense of more complex and less interpretable activation patterns. These findings advance the mechanistic interpretability of domain-specific protein language models and suggest that, while TopK SAEs are sufficient for mapping latent features to concepts, Ordered SAEs are preferable when precise generative steering is required.

---

## 6. FNOPT: Resolution-Agnostic, Self-Supervised Cloth Simulation using Meta-Optimization with Fourier Neural Operators

**论文链接:** [http://arxiv.org/abs/2512.05762v1](http://arxiv.org/abs/2512.05762v1)

**作者:** Ruochen Chen, Thuy Tran, Shaifali Parashar

**发布时间:** 2025-12-05

**备注:** Accepted for WACV

### GPT解析

### 总结

FNOpt是一种自监督布料模拟框架，将时间积分表述为优化问题，并使用傅里叶神经算子参数化的分辨率无关神经优化器进行训练。

### 背景

之前的神经模拟器通常依赖大量真实数据或牺牲细节，且在不同分辨率和运动模式上泛化能力差。

### 目的

开发一种能够在不同分辨率和运动模式下稳定准确模拟布料动力学的方法，减少对精心策划数据的需求。

### 方法

使用傅里叶神经算子参数化神经优化器，仅在粗网格上基于物理损失进行训练，学习模拟物理合理的布料动力学。

### 主要发现

FNOpt无需重新训练即可泛化到更精细的分辨率，能够捕捉精细褶皱并保持模拟稳定性，在分布外设置中的准确性和鲁棒性均优于先前基于学习的方法。

### 结论

基于FNO的元优化是布料神经模拟器的有力替代方案，减少了对精心策划数据的需求，提高了跨分辨率可靠性。

### 翻译

我们提出了FNOpt，一种自监督布料模拟框架，将时间积分表述为优化问题，并训练一个由傅里叶神经算子参数化的分辨率无关神经优化器。先前的神经模拟器通常依赖大量真实数据或牺牲细节，且在不同分辨率和运动模式上泛化能力差。相比之下，FNOpt学习模拟物理合理的布料动力学，能够在不同网格分辨率和运动模式下实现稳定准确的模拟而无需重新训练。仅在粗网格上基于物理损失进行训练，FNOpt能够泛化到更精细的分辨率，捕捉精细褶皱并保持模拟稳定性。在基准布料模拟数据集上的广泛评估表明，FNOpt在分布外设置中的准确性和鲁棒性均优于先前基于学习的方法。这些结果将基于FNO的元优化定位为先前布料神经模拟器的有力替代方案，从而减少了对精心策划数据的需求并提高了跨分辨率可靠性。


### 论文摘要

We present FNOpt, a self-supervised cloth simulation framework that formulates time integration as an optimization problem and trains a resolution-agnostic neural optimizer parameterized by a Fourier neural operator (FNO). Prior neural simulators often rely on extensive ground truth data or sacrifice fine-scale detail, and generalize poorly across resolutions and motion patterns. In contrast, FNOpt learns to simulate physically plausible cloth dynamics and achieves stable and accurate rollouts across diverse mesh resolutions and motion patterns without retraining. Trained only on a coarse grid with physics-based losses, FNOpt generalizes to finer resolutions, capturing fine-scale wrinkles and preserving rollout stability. Extensive evaluations on a benchmark cloth simulation dataset demonstrate that FNOpt outperforms prior learning-based approaches in out-of-distribution settings in both accuracy and robustness. These results position FNO-based meta-optimization as a compelling alternative to previous neural simulators for cloth, thus reducing the need for curated data and improving cross-resolution reliability.

---

## 7. 论文ID: 2512.05695v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.05695v1.json'

---

## 8. 论文ID: 2512.05680v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.05680v1.json'

---

## 9. The Power of Network Pluralism: Multi-Perspective Modeling of Heterogeneous Legal Document Networks

**论文链接:** [http://arxiv.org/abs/2512.05679v1](http://arxiv.org/abs/2512.05679v1)

**作者:** Titus Pünder, Corinna Coupette

**发布时间:** 2025-12-05

**备注:** 35 pages, 15 figures

### GPT解析

### 总结

本文提出'网络多元化'概念框架，强调在复杂系统研究中需考虑多种视角以获得更完整、更有意义和更稳健的结果。通过分析复杂法律系统展示了多网络分析的优势，并说明了如何通过多视角增强研究深度和透明度。

### 背景

洞见是相对的，受假设、范围或方法等多种因素影响，单一视角无法产生完整知识。认识论多元化要求研究人员同时考虑多种视角，以获得对研究现象的整体理解。

### 目的

将认识论多元化原则转化为网络科学领域，引入网络多元化框架，利用多视角方法产生更完整、更有意义和更稳健的结果，并在复杂系统分析中展示其优势。

### 方法

通过分析复杂法律系统构建网络空间，包括不同政府部门的文档引用和组织层次结构。利用异质性进行多网络分析，展示互补视角如何解释高层次发现，比较不同网络如何通过差异学习，以及将指标与视角关联如何提高分析结果的透明度和稳健性。

### 主要发现

互补视角可帮助解释高层次发现；比较来自相同数据的不同网络使研究人员能通过差异学习；将指标与视角关联可提高网络分析结果的透明度和稳健度；分析网络空间需将领域变异维度映射到网络建模决策和度量参数。

### 结论

将变异维度映射到网络建模决策和度量参数是具有挑战性的跨学科任务，但本研究作为蓝图，促进了网络多元化在领域驱动网络研究中的更广泛应用。

### 翻译

洞见是相对的 - 受一系列因素影响，如假设、范围或方法，这些因素共同定义了研究视角。在规范性和经验性领域，这一洞见导致了单一视角无法产生完整知识的结论。作为回应，认识论多元化要求研究人员同时考虑多种视角，以获得对所研究现象的整体理解。将这一要求转化为网络科学，我们的工作引入了网络多元化作为概念框架，利用多视角性产生更完整、更有意义和更稳健的结果。我们通过分析复杂法律系统的实际案例开发和展示了这种方法的优势，构建了一个来自不同政府部门的文档引用网络空间，同时包括文档层级之上的组织层次结构和文档层级之下的精细结构。利用由此产生的异质性进行多网络分析，我们展示了互补视角如何帮助解释否则会显得高层次的发现，比较从相同数据导出的几个网络如何使研究人员通过差异学习，以及将指标与视角联系起来如何提高网络分析结果的透明度和稳健性。要分析作为视角的网络空间，研究人员需要将给定领域的变异维度映射到网络建模决策和网络度量参数。虽然这仍然是一个具有挑战性和本质上跨学科的任务，但我们的工作作为蓝图，促进了网络多元化在领域驱动的网络研究中的更广泛采用。


### 论文摘要

Insights are relative - influenced by a range of factors such as assumptions, scopes, or methods that together define a research perspective. In normative and empirical fields alike, this insight has led to the conclusion that no single perspective can generate complete knowledge. As a response, epistemological pluralism mandates that researchers consider multiple perspectives simultaneously to obtain a holistic understanding of their phenomenon under study. Translating this mandate to network science, our work introduces Network Pluralism as a conceptual framework that leverages multi-perspectivity to yield more complete, meaningful, and robust results. We develop and demonstrate the benefits of this approach via a hands-on analysis of complex legal systems, constructing a network space from references across documents from different branches of government as well as including organizational hierarchy above and fine-grained structure below the document level. Leveraging the resulting heterogeneity in a multi-network analysis, we show how complementing perspectives can help contextualize otherwise high-level findings, how contrasting several networks derived from the same data enables researchers to learn by difference, and how relating metrics to perspectives may increase the transparency and robustness of network-analytical results. To analyze a space of networks as perspectives, researchers need to map dimensions of variation in a given domain to network-modeling decisions and network-metric parameters. While this remains a challenging and inherently interdisciplinary task, our work acts as a blueprint to facilitate the broader adoption of Network Pluralism in domain-driven network research.

---

## 10. Self-Supervised AI-Generated Image Detection: A Camera Metadata Perspective

**论文链接:** [http://arxiv.org/abs/2512.05651v1](http://arxiv.org/abs/2512.05651v1)

**作者:** Nan Zhong, Mian Zou, Yiran Xu, Zhenxing Qian, Xinpeng Zhang, Baoyuan Wu, Kede Ma

**发布时间:** 2025-12-05

### GPT解析

### 总结

研究提出了一种基于EXIF标签的自监督方法来检测AI生成图像，利用相机元数据学习数字摄影特征，通过高斯混合模型和分类器实现检测，实验证明该方法在多种生成模型上表现优异，具有良好的泛化能力和鲁棒性。

### 背景

AI生成图像的激增给多媒体取证带来了日益严峻的挑战，许多现有检测器依赖于对特定生成模型内部机制的假设，限制了它们的跨模型适用性。

### 目的

引入一种自监督方法来检测AI生成的图像，利用相机元数据学习数字摄影固有的特征，提高检测器的跨模型适用性。

### 方法

通过分类和排序EXIF标签训练特征提取器；使用高斯混合模型对摄影图像分布建模进行一类检测；扩展到二元检测，将提取器视为分类器的正则化器，在空间打乱块的高频残差上运行。

### 主要发现

在各种生成模型上的实验表明，EXIF诱导检测器显著推动了最先进技术的发展，对野外样本提供了强大的泛化能力，并对常见的良性图像扰动具有鲁棒性。

### 结论

基于EXIF标签的自监督方法是检测AI生成图像的有效方法，具有良好的跨模型适用性和鲁棒性。

### 翻译

AI生成图像的激增给多媒体取证带来了日益严峻的挑战，然而许多现有检测器依赖于对特定生成模型内部机制的假设，限制了它们的跨模型适用性。我们介绍了一种用于检测AI生成图像的自监督方法，它利用相机元数据——特别是可交换图像文件格式（EXIF）标签——来学习数字摄影固有的特征。我们的预训练任务通过分类分类别的EXIF标签（例如相机型号和场景类型）和对序数和连续EXIF标签（例如焦距和光圈值）进行成对排序，仅使用相机拍摄的图像来训练特征提取器。使用这些EXIF诱导的特征，我们首先通过高斯混合模型对摄影图像的分布进行建模，将低可能性样本标记为AI生成来执行一类检测。然后我们扩展到二元检测，将学习到的提取器视为同一架构分类器的强正则化器，在空间打乱块的高频残差上运行。在各种生成模型上的大量实验表明，我们的EXIF诱导检测器显著推动了最先进技术的发展，对野外样本提供了强大的泛化能力，并对常见的良性图像扰动具有鲁棒性。


### 论文摘要

The proliferation of AI-generated imagery poses escalating challenges for multimedia forensics, yet many existing detectors depend on assumptions about the internals of specific generative models, limiting their cross-model applicability. We introduce a self-supervised approach for detecting AI-generated images that leverages camera metadata -- specifically exchangeable image file format (EXIF) tags -- to learn features intrinsic to digital photography. Our pretext task trains a feature extractor solely on camera-captured photographs by classifying categorical EXIF tags (\eg, camera model and scene type) and pairwise-ranking ordinal and continuous EXIF tags (\eg, focal length and aperture value). Using these EXIF-induced features, we first perform one-class detection by modeling the distribution of photographic images with a Gaussian mixture model and flagging low-likelihood samples as AI-generated. We then extend to binary detection that treats the learned extractor as a strong regularizer for a classifier of the same architecture, operating on high-frequency residuals from spatially scrambled patches. Extensive experiments across various generative models demonstrate that our EXIF-induced detectors substantially advance the state of the art, delivering strong generalization to in-the-wild samples and robustness to common benign image perturbations.

---

## 11. 论文ID: 2512.05638v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.05638v1.json'

---

## 12. MedDIFT: Multi-Scale Diffusion-Based Correspondence in 3D Medical Imaging

**论文链接:** [http://arxiv.org/abs/2512.05571v1](http://arxiv.org/abs/2512.05571v1)

**作者:** Xingyu Zhang, Anna Reithmeir, Fryderyk Kögl, Rickmer Braren, Julia A. Schnabel, Daniel M. Lang

**发布时间:** 2025-12-05

### GPT解析

### 总结

MedDIFT是一种创新的无需训练的3D医学图像对应框架，利用扩散模型的多尺度特征作为体素描述符，实现了与学习型模型相当的配准精度，同时超越了传统方法。

### 背景

医学图像之间的精确空间对应关系对于纵向分析、病变跟踪和图像引导干预至关重要。现有医学图像配准方法依赖于基于局部相似度的强度度量，这些方法无法捕捉全局语义结构，且在低对比度或解剖学变异区域经常产生误匹配。

### 目的

开发一种能够捕捉全局语义结构的医学图像配准方法，解决现有方法在低对比度或解剖学变异区域的问题。

### 方法

提出MedDIFT框架，利用预训练的潜在医学扩散模型的多尺度特征作为体素描述符，将扩散激活融合成丰富的体素级描述符，并通过余弦相似性进行匹配，同时支持可选的局部搜索先验。

### 主要发现

在公开的肺CT数据集上，MedDIFT实现了与最先进的学习型UniGradICON模型相当的对应精度，超越了传统的基于B样条的配准方法，且不需要任何特定任务模型训练。消融实验证实多级特征融合和适度的扩散噪声可以提高性能。

### 结论

扩散模型的中间表示编码了丰富的几何和语义信息，MedDIFT框架能够有效利用这些信息进行医学图像配准，在无需训练的情况下达到了最先进性能。

### 翻译

医学图像之间的精确空间对应关系对于纵向分析、病变跟踪和图像引导干预至关重要。医学图像配准方法依赖于基于局部相似度的强度度量，这些方法无法捕捉全局语义结构，且在低对比度或解剖学变异区域经常产生误匹配。扩散模型的最新进展表明其中间表示编码了丰富的几何和语义信息。我们提出了MedDIFT，一种无需训练的3D对应框架，它利用预训练的潜在医学扩散模型的多尺度特征作为体素描述符。MedDIFT将扩散激活融合成丰富的体素级描述符，并通过余弦相似性进行匹配，并带有可选的局部搜索先验。在公开的肺CT数据集上，MedDIFT实现了与最先进的学习型UniGradICON模型相当的对应精度，并超越了传统的基于B样条的配准方法，无需任何特定任务的模型训练。消融实验证实多级特征融合和适度的扩散噪声可以提高性能。


### 论文摘要

Accurate spatial correspondence between medical images is essential for longitudinal analysis, lesion tracking, and image-guided interventions. Medical image registration methods rely on local intensity-based similarity measures, which fail to capture global semantic structure and often yield mismatches in low-contrast or anatomically variable regions. Recent advances in diffusion models suggest that their intermediate representations encode rich geometric and semantic information. We present MedDIFT, a training-free 3D correspondence framework that leverages multi-scale features from a pretrained latent medical diffusion model as voxel descriptors. MedDIFT fuses diffusion activations into rich voxel-wise descriptors and matches them via cosine similarity, with an optional local-search prior. On a publicly available lung CT dataset, MedDIFT achieves correspondence accuracy comparable to the state-of-the-art learning-based UniGradICON model and surpasses conventional B-spline-based registration, without requiring any task-specific model training. Ablation experiments confirm that multi-level feature fusion and modest diffusion noise improve performance.

---

## 13. 论文ID: 2512.05568v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.05568v1.json'

---

## 14. 论文ID: 2512.05530v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.05530v1.json'

---

## 15. DashFusion: Dual-stream Alignment with Hierarchical Bottleneck Fusion for Multimodal Sentiment Analysis

**论文链接:** [http://arxiv.org/abs/2512.05515v1](http://arxiv.org/abs/2512.05515v1)

**作者:** Yuhua Wen, Qifei Li, Yingying Zhou, Yingming Gao, Zhengqi Wen, Jianhua Tao, Ya Li

**发布时间:** 2025-12-05

**备注:** Accepted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2025

### GPT解析

### 总结

该论文提出了名为DashFusion的新型多模态情感分析框架，通过双流对齐和分层瓶颈融合技术解决了多模态对齐和融合的挑战，在多个数据集上实现了最先进的性能。

### 背景

多模态情感分析整合文本、图像和音频等多种模态以提供更全面的情感理解，但面临对齐和融合两大挑战。现有方法通常单独处理对齐或融合，导致性能和效率受限。

### 目的

开发一种能够同时解决多模态情感分析中对齐和融合问题的框架，提高性能和计算效率。

### 方法

提出DashFusion框架，包含双流对齐模块（通过时间对齐和语义对齐同步多模态特征）、监督式对比学习（利用标签信息改进模态特征）和分层瓶颈融合（通过压缩的瓶颈令牌逐步整合多模态信息）。

### 主要发现

在CMU-MOSI、CMU-MOSEI和CH-SIMS三个数据集上的实验结果表明，DashFusion在各种指标上达到了最先进的性能，消融研究证实了对齐和融合技术的有效性。

### 结论

DashFusion框架有效解决了多模态情感分析中的对齐和融合挑战，在性能和效率方面都有优势，相关代码已在GitHub上公开。

### 翻译

多模态情感分析整合文本、图像和音频等多种模态，以提供更全面的情感理解。然而，有效的多模态情感分析面临对齐和融合的挑战。对齐需要跨模态同步时间和语义信息，而融合需要将已对齐的特征整合为统一表示。现有方法通常单独处理对齐或融合，导致性能和效率受限。为了解决这些问题，我们提出了一种名为双流对齐与分层瓶颈融合（DashFusion）的新框架。首先，双流对齐模块通过时间和语义对齐同步多模态特征。时间对齐使用跨模态注意力建立多模态序列之间的帧级对应关系，语义对齐通过对比学习确保特征空间的一致性。其次，监督式对比学习利用标签信息来改进模态特征。最后，分层瓶颈融合通过压缩的瓶颈令牌逐步整合多模态信息，在性能和计算效率之间取得平衡。我们在CMU-MOSI、CMU-MOSEI和CH-SIMS三个数据集上评估DashFusion。实验结果表明，DashFusion在各种指标上达到了最先进的性能，消融研究证实了我们对齐和融合技术的有效性。我们的实验代码可在https://github.com/ultramarineX/DashFusion获取。


### 论文摘要

Multimodal sentiment analysis (MSA) integrates various modalities, such as text, image, and audio, to provide a more comprehensive understanding of sentiment. However, effective MSA is challenged by alignment and fusion issues. Alignment requires synchronizing both temporal and semantic information across modalities, while fusion involves integrating these aligned features into a unified representation. Existing methods often address alignment or fusion in isolation, leading to limitations in performance and efficiency. To tackle these issues, we propose a novel framework called Dual-stream Alignment with Hierarchical Bottleneck Fusion (DashFusion). Firstly, dual-stream alignment module synchronizes multimodal features through temporal and semantic alignment. Temporal alignment employs cross-modal attention to establish frame-level correspondences among multimodal sequences. Semantic alignment ensures consistency across the feature space through contrastive learning. Secondly, supervised contrastive learning leverages label information to refine the modality features. Finally, hierarchical bottleneck fusion progressively integrates multimodal information through compressed bottleneck tokens, which achieves a balance between performance and computational efficiency. We evaluate DashFusion on three datasets: CMU-MOSI, CMU-MOSEI, and CH-SIMS. Experimental results demonstrate that DashFusion achieves state-of-the-art performance across various metrics, and ablation studies confirm the effectiveness of our alignment and fusion techniques. The codes for our experiments are available at https://github.com/ultramarineX/DashFusion.

---

## 16. UniFS: Unified Multi-Contrast MRI Reconstruction via Frequency-Spatial Fusion

**论文链接:** [http://arxiv.org/abs/2512.05481v1](http://arxiv.org/abs/2512.05481v1)

**作者:** Jialin Li, Yiwei Ren, Kai Pan, Dong Wei, Pujin Cheng, Xian Wu, Xiaoying Tang

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种名为UniFS的统一频率-空间融合模型，用于解决多对比度MR重建中现有方法难以推广到不同k空间欠采样模式的问题。该模型整合了三个关键模块，能够处理多种k空间欠采样模式而无需重新训练，并通过融合跨模态频率信息显著提高了重建质量。

### 背景

多对比度MR重建(MCMR)是一个热门研究领域，利用高质量辅助模态重建目标模态。然而，现有方法难以推广到不同k空间欠采样模式，需要为每种模式单独训练模型，限制了实用性。此外，现有方法往往只关注空间信息而忽略频率特性，或只提取浅层频率特征，无法充分利用跨模态的互补频率信息。

### 目的

开发一种统一的频率-空间融合模型，能够处理多种k空间欠采样模式而无需重新训练，并充分利用跨模态频率信息提高重建质量。

### 方法

UniFS整合了三个关键模块：1)跨模态频率融合模块；2)基于自适应掩码的提示学习模块；3)双分支互补细化模块。这些模块共同工作，从各种k空间欠采样模式中提取域不变特征，同时动态适应各自的变异。此外，引入自适应提示引导的频率融合模块进行k空间学习，显著提高模型泛化性能。

### 主要发现

在BraTS和HCP数据集上使用各种k空间欠采样模式和加速因子(包括未见过的模式)评估了UniFS。实验结果表明，UniFS在多种场景下实现了最先进的性能，证明了其优秀的泛化能力和重建质量。

### 结论

UniFS成功解决了现有MCMR方法难以推广到不同k空间欠采样模式的问题，通过融合跨模态频率信息显著提高了重建质量。该模型无需为每种欠采样模式重新训练，提高了实用性和适用性。代码已公开在https://github.com/LIKP0/UniFS。

### 翻译

最近，多对比度MR重建(MCMR)已成为一个热门研究领域，它利用高质量的辅助模态来重建感兴趣的目标模态。然而，现有方法往往难以推广到不同的k空间欠采样模式，需要为每种特定模式单独训练模型，这限制了它们的实际适用性。为了解决这一挑战，我们提出了UniFS，一种统一的频率-空间融合模型，专为处理MCMR任务中的多种k空间欠采样模式而设计，无需任何重新训练。UniFS集成了三个关键模块：跨模态频率融合模块、基于自适应掩码的提示学习模块和双分支互补细化模块。这些模块协同工作，从各种k空间欠采样模式中提取域不变特征，同时动态适应各自的变异。现有MCMR方法的另一个局限性是它们倾向于只关注空间信息而忽略频率特性，或者只提取浅层频率特征，因此无法充分利用互补的跨模态频率信息。为了缓解这一问题，UniFS引入了用于k空间学习的自适应提示引导频率融合模块，显著提高了模型的泛化性能。我们在BraTS和HCP数据集上使用各种k空间欠采样模式和加速因子(包括以前未见过的模式)评估了我们的模型，以全面评估UniFS的泛化能力。多种场景下的实验结果表明，UniFS实现了最先进的性能。我们的代码可在https://github.com/LIKP0/UniFS获取。


### 论文摘要

Recently, Multi-Contrast MR Reconstruction (MCMR) has emerged as a hot research topic that leverages high-quality auxiliary modalities to reconstruct undersampled target modalities of interest. However, existing methods often struggle to generalize across different k-space undersampling patterns, requiring the training of a separate model for each specific pattern, which limits their practical applicability. To address this challenge, we propose UniFS, a Unified Frequency-Spatial Fusion model designed to handle multiple k-space undersampling patterns for MCMR tasks without any need for retraining. UniFS integrates three key modules: a Cross-Modal Frequency Fusion module, an Adaptive Mask-Based Prompt Learning module, and a Dual-Branch Complementary Refinement module. These modules work together to extract domain-invariant features from diverse k-space undersampling patterns while dynamically adapt to their own variations. Another limitation of existing MCMR methods is their tendency to focus solely on spatial information while neglect frequency characteristics, or extract only shallow frequency features, thus failing to fully leverage complementary cross-modal frequency information. To relieve this issue, UniFS introduces an adaptive prompt-guided frequency fusion module for k-space learning, significantly enhancing the model's generalization performance. We evaluate our model on the BraTS and HCP datasets with various k-space undersampling patterns and acceleration factors, including previously unseen patterns, to comprehensively assess UniFS's generalizability. Experimental results across multiple scenarios demonstrate that UniFS achieves state-of-the-art performance. Our code is available at https://github.com/LIKP0/UniFS.

---

## 17. Learning from Self Critique and Refinement for Faithful LLM Summarization

**论文链接:** [http://arxiv.org/abs/2512.05387v1](http://arxiv.org/abs/2512.05387v1)

**作者:** Ting-Yao Hu, Hema Swetha Koppula, Hadi Pouransari, Cem Koc, Oncel Tuzel, Raviteja Vemulapalli

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种名为SCRPO（基于自我批判和精炼的偏好优化）的方法，用于减少大型语言模型在长文本生成任务中的幻觉问题。

### 背景

大型语言模型在执行摘要等长文本生成任务时，经常会产生与输入内容不相关的幻觉内容。先前的研究表明，通过迭代批判和精炼先前生成的输出来减少幻觉，但这些方法需要额外的测试时间计算或假设可以访问更强大的教师模型，导致成本高昂且实用性较低。

### 目的

开发一种自我监督训练框架，利用大型语言模型自身的批判和精炼能力构建偏好数据集，并通过偏好学习改进同一模型，以实现忠实摘要。

### 方法

SCRPO是一种自我监督训练框架，首先利用大型语言模型自身的批判和精炼能力构建偏好数据集，然后应用偏好学习来改进同一模型以实现忠实摘要。

### 主要发现

在三个摘要基准测试（XSUM、CNNDM和SAMSum）上的实验表明，该方法在忠实性指标方面优于最先进的自监督学习方法，同时保持或改善了衡量摘要整体质量的其他指标。与测试时精炼相比，该方法不仅提高了效率，还产生了更忠实的摘要。

### 结论

SCRPO方法能够有效地减少大型语言模型在摘要任务中的幻觉问题，同时保持或提高摘要的整体质量，并且比现有的测试时精炼方法更高效。

### 翻译

大型语言模型在执行摘要等长文本生成任务时，经常会产生与输入内容不相关的幻觉内容。先前的研究表明，通过迭代批判和精炼先前生成的输出来减少幻觉，但这些方法需要额外的测试时间计算或假设可以访问更强大的教师模型，导致它们成本高昂且实用性较低。在这项工作中，我们提出了基于自我批判和精炼的偏好优化（SCRPO），这是一种自我监督训练框架，首先利用大型语言模型自身的批判和精炼能力构建偏好数据集，然后应用偏好学习来改进同一模型以实现忠实摘要。在三个摘要基准测试（XSUM、CNNDM和SAMSum）上的实验表明，我们的方法在忠实性指标方面优于最先进的自监督学习方法，同时保持或改善了衡量摘要整体质量的其他指标。此外，与测试时精炼相比，我们的方法不仅提高了效率，还产生了更忠实的摘要。


### 论文摘要

Large Language Models (LLMs) often suffer from hallucinations: output content that is not grounded in the input context, when performing long-form text generation tasks such as summarization. Prior works have shown that hallucinations can be reduced by iteratively critiquing and refining previously generated outputs using either the same model or a more powerful teacher model as the critique. However, these approaches either require additional test-time compute or assume access to more powerful teacher models, making them costly and less practical. In this work, we propose Self Critique and Refinement-based Preference Optimization (SCRPO), which is a self-supervised training framework that first constructs a preference dataset by leveraging the LLM's own critique and refinement capabilities, and then applies preference learning to improve the same LLM for faithful summarization. Experiments on three summarization benchmarks (XSUM CNNDM and SAMSum), demonstrate that our approach outperforms state-of-the-art self-supervised learning methods in terms of faithfulness metrics while either maintaining or improving other metrics that measure the overall quality of the summary. Moreover, compared to test-time refinement, our approach not only improves efficiency but also results in more faithful summaries.

---

## 18. Generalization Beyond Benchmarks: Evaluating Learnable Protein-Ligand Scoring Functions on Unseen Targets

**论文链接:** [http://arxiv.org/abs/2512.05386v1](http://arxiv.org/abs/2512.05386v1)

**作者:** Jakub Kopko, David Graber, Saltuk Mustafa Eyrilmez, Stanislav Mazurenko, David Bednar, Jiri Sedlar, Josef Sivic

**发布时间:** 2025-12-05

**备注:** 15 pages, 6 figures, submitted to NeurIPS 2025 AI4Science Workshop

### GPT解析

### 总结

本研究评估了机器学习在分子设计中蛋白质-配体评分函数的泛化能力，并探讨了提高其在新靶点上预测性能的方法。

### 背景

随着机器学习在分子设计中变得越来越重要，确保可学习的蛋白质-配体评分函数在新蛋白质靶点上的可靠性至关重要。

### 目的

评估最先进的评分函数在模拟具有有限数量已知结构和实验亲和力测量数据的靶点数据集分割上的泛化能力。

### 方法

分析常用基准测试是否反映了泛化到新靶点的真正挑战；研究大规模自监督预训练是否能弥合泛化差距；探索利用有限的测试靶点数据提高评分函数性能的简单方法。

### 主要发现

常用的基准测试不能反映泛化到新靶点的真正挑战；大规模自监督预训练有可能弥合这一泛化差距；利用有限的测试靶点数据可以提高评分函数性能。

### 结论

需要更严格的评估协议；为设计具有预测能力延伸到新蛋白质靶点的评分函数提供实际指导。

### 翻译

随着机器学习在分子设计中变得越来越重要，确保可学习的蛋白质-配体评分函数在新蛋白质靶点上的可靠性至关重要。虽然许多评分函数在标准基准测试中表现良好，但它们在训练数据之外泛化的能力仍然是一个重大挑战。在本工作中，我们评估了最先进的评分函数在模拟具有有限数量已知结构和实验亲和力测量数据的靶点数据集分割上的泛化能力。我们的分析表明，常用的基准测试不能反映泛化到新靶点的真正挑战。我们还研究了大规模自监督预训练是否能弥合这一泛化差距，并提供了其潜在可能性的初步证据。此外，我们探究了利用有限的测试靶点数据来提高评分函数性能的简单方法。我们的发现强调了需要更严格的评估协议，并为设计具有预测能力延伸到新蛋白质靶点的评分函数提供了实用指导。


### 论文摘要

As machine learning becomes increasingly central to molecular design, it is vital to ensure the reliability of learnable protein-ligand scoring functions on novel protein targets. While many scoring functions perform well on standard benchmarks, their ability to generalize beyond training data remains a significant challenge. In this work, we evaluate the generalization capability of state-of-the-art scoring functions on dataset splits that simulate evaluation on targets with a limited number of known structures and experimental affinity measurements. Our analysis reveals that the commonly used benchmarks do not reflect the true challenge of generalizing to novel targets. We also investigate whether large-scale self-supervised pretraining can bridge this generalization gap and we provide preliminary evidence of its potential. Furthermore, we probe the efficacy of simple methods that leverage limited test-target data to improve scoring function performance. Our findings underscore the need for more rigorous evaluation protocols and offer practical guidance for designing scoring functions with predictive power extending to novel protein targets.

---

## 19. 论文ID: 2512.05377v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.05377v1.json'

---

## 20. Hypothesis-Based Particle Detection for Accurate Nanoparticle Counting and Digital Diagnostics

**论文链接:** [http://arxiv.org/abs/2512.05346v1](http://arxiv.org/abs/2512.05346v1)

**作者:** Neil H. Kim, Xiao-Liu Chu, Joseph B. DeGrandchamp, Matthew R. Foreman

**发布时间:** 2025-12-05

**备注:** Main text (14 pages, 5 figures, 1 table) and supplementary information (5 pages, 3 figures, 2 tables). Supporting code at https://github.com/Optical-Theory-Group/Hypothesis-Test-Based-Particle-Detection

### GPT解析

### 总结

该论文介绍了一种基于纳米粒子的成像检测颗粒计数算法，采用多假设统计检验框架，无需训练数据或参数调整，在多种条件下表现出稳健的计数准确性，并成功应用于SARS-CoV-2 DNA生物标志物的检测。

### 背景

数字检测技术代表传统诊断方法的转变，通过离散计数生物分子报告因子，能够精确检测低丰度分析物，这对早期疾病诊断和个性化医学至关重要。

### 目的

开发一种基于纳米粒子的成像检测颗粒计数算法，该方法不需要训练数据或经验参数调整，在各种条件下保持稳健的计数准确性，并能够解释计数结果。

### 方法

提出一种表述为多假设统计检验的颗粒计数算法，使用明确的成像形成模型和惩罚似然规则进行评估。通过数值模拟评估算法在弱信号、可变背景、放大变化和中等PSF不匹配情况下的性能，并在实验暗场图像中应用该算法检测SARS-CoV-2 DNA生物标志物。

### 主要发现

算法在各种条件下表现出稳健的计数准确性；颗粒可分辨性测试揭示了特征错误模式，包括在非常小距离下的欠计数和在分辨率极限附近的局部过计数；在对照样本和阳性样本之间观察到颗粒计数分布的统计学显著差异；完整计数统计表现出一致的过度离散，为非特异性靶向诱导的颗粒聚集提供了见解。

### 结论

该方法确立了作为数字分子诊断中基于纳米粒子的检测的可靠框架，不需要训练数据或参数调整，输出结果可解释，并在实际应用中表现出色。

### 翻译

数字检测技术代表了传统诊断方法的转变，通过离散计数生物分子报告因子，能够精确检测低丰度分析物，这对早期疾病诊断和个性化医学至关重要。在这种范式下，我们提出了一种基于纳米粒子的成像检测的颗粒计数算法，将其表述为明确的成像形成模型下的多假设统计检验，并使用惩罚似然规则进行评估。与阈值化或机器学习方法相比，这种方法不需要训练数据或经验参数调整，其输出通过与成像物理学和统计决策理论的直接联系保持可解释性。通过数值模拟，我们证明了该方法在弱信号、可变背景、放大变化和中等PSF不匹配情况下的稳健计数准确性。颗粒可分辨性测试进一步揭示了特征错误模式，包括在非常小距离下的欠计数和在分辨率极限附近的局部过计数。实际上，我们还通过应用包含基于纳米粒子的检测SARS-CoV-2 DNA生物标志物的实验暗场图像，确认了该算法的实用性。在对照样本和阳性样本之间观察到颗粒计数分布的统计学显著差异。获得的完整计数统计进一步表现出一致的过度离散，并为非特异性靶向诱导的颗粒聚集提供了见解。这些结果确立了我们的方法作为数字分子诊断中基于纳米粒子的检测的可靠框架。


### 论文摘要

Digital assays represent a shift from traditional diagnostics and enable the precise detection of low-abundance analytes, critical for early disease diagnosis and personalized medicine, through discrete counting of biomolecular reporters. Within this paradigm, we present a particle counting algorithm for nanoparticle based imaging assays, formulated as a multiple-hypothesis statistical test under an explicit image-formation model and evaluated using a penalized likelihood rule. In contrast to thresholding or machine learning methods, this approach requires no training data or empirical parameter tuning, and its outputs remain interpretable through direct links to imaging physics and statistical decision theory.   Through numerical simulations we demonstrate robust count accuracy across weak signals, variable backgrounds, magnification changes and moderate PSF mismatch. Particle resolvability tests further reveal characteristic error modes, including under-counting at very small separations and localized over-counting near the resolution limit. Practically, we also confirm the algorithm's utility, through application to experimental dark-field images comprising a nanoparticle-based assay for detection of DNA biomarkers derived from SARS-CoV-2. Statistically significant differences in particle count distributions are observed between control and positive samples. Full count statistics obtained further exhibit consistent over-dispersion, and provide insight into non-specific and target-induced particle aggregation. These results establish our method as a reliable framework for nanoparticle-based detection assays in digital molecular diagnostics.

---

## 21. Competition, stability, and functionality in excitatory-inhibitory neural circuits

**论文链接:** [http://arxiv.org/abs/2512.05252v1](http://arxiv.org/abs/2512.05252v1)

**作者:** Simone Betteti, William Retnaraj, Alexander Davydov, Jorge Cortés, Francesco Bullo

**发布时间:** 2025-12-04

### GPT解析

### 总结

该研究将能量框架扩展到不对称神经系统中，揭示了神经动力学的博弈论结构，并利用网络理论原理研究E-I网络的稳定性，重新审视了理论神经科学中的标准模型。

### 背景

基于能量的模型已成为理解理论神经科学和机器学习中计算与稳定性的中心范式，但传统的能量框架依赖于突触或权重矩阵的对称性，这一约束排除了生物真实的系统如兴奋性-抑制性(E-I)网络。

### 目的

扩展能量框架以适应不对称神经发放率网络，研究E-I网络中神经活动的调节和平衡，并重新审视理论神经科学中的标准框架。

### 方法

将能量框架扩展到不对称发放率网络，利用网络理论的稳定性原理研究E-I网络，结合博弈-能量解释分析标准神经科学模型。

### 主要发现

神经动力学具有潜在的博弈论结构，每个神经元作为一个代理寻求最小化自身能量；侧抑制微回路的皮层柱可以作为对比增强器，通过分层兴奋-抑制相互作用选择性地增强环境中的细微差异。

### 结论

研究结果桥接了神经计算的能量观和博弈论观点，为系统性构建生物基础且动态稳定的神经架构提供了途径。

### 翻译

基于能量的模型已成为理解理论神经科学和机器学习中计算与稳定性的中心范式。然而，能量框架通常依赖于突触或权重矩阵的对称性 - 这一约束排除了生物真实的系统如兴奋性-抑制性(E-I)网络。当对称性被放松时，全局能量景观的经典概念失效，使得不对称神经系统的动力学概念上失去锚点。在这项工作中，我们将能量框架扩展到不对称发放率网络，揭示了神经动力学中潜在的博弈论结构，其中每个神经元都是一个寻求最小化自身能量的代理。此外，我们利用网络理论的严格稳定性原理来研究E-I网络中神经活动的调节和平衡。我们将这种新的博弈-能量解释和稳定性结果重新审视理论神经科学中的标准框架，如Wilson-Cowan模型和侧抑制模型。这些见解使我们能够将侧抑制微回路的皮层柱作为对比增强器进行研究 - 通过分层兴奋-抑制相互作用选择性地增强环境中细微差异。我们的结果桥接了神经计算的能量观和博弈论观点，为系统性构建生物基础且动态稳定的神经架构提供了途径。


### 论文摘要

Energy-based models have become a central paradigm for understanding computation and stability in both theoretical neuroscience and machine learning. However, the energetic framework typically relies on symmetry in synaptic or weight matrices - a constraint that excludes biologically realistic systems such as excitatory-inhibitory (E-I) networks. When symmetry is relaxed, the classical notion of a global energy landscape fails, leaving the dynamics of asymmetric neural systems conceptually unanchored. In this work, we extend the energetic framework to asymmetric firing rate networks, revealing an underlying game-theoretic structure for the neural dynamics in which each neuron is an agent that seeks to minimize its own energy. In addition, we exploit rigorous stability principles from network theory to study regulation and balancing of neural activity in E-I networks. We combine the novel game-energetic interpretation and the stability results to revisit standard frameworks in theoretical neuroscience, such as the Wilson-Cowan and lateral inhibition models. These insights allow us to study cortical columns of lateral inhibition microcircuits as contrast enhancer - with the ability to selectively sharpen subtle differences in the environment through hierarchical excitation-inhibition interplay. Our results bridge energetic and game-theoretic views of neural computation, offering a pathway toward the systematic engineering of biologically grounded, dynamically stable neural architectures.

---

## 22. 论文ID: 2512.05234v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.05234v1.json'

---

## 23. Mitigating the Antigenic Data Bottleneck: Semi-supervised Learning with Protein Language Models for Influenza A Surveillance

**论文链接:** [http://arxiv.org/abs/2512.05222v1](http://arxiv.org/abs/2512.05222v1)

**作者:** Yanhua Xu

**发布时间:** 2025-12-04

**备注:** V0: initial draft uploaded

### GPT解析

### 总结

该研究探讨了将预训练蛋白质语言模型与半监督学习结合，用于预测流感A病毒抗原性的方法，解决了传统HI检测劳动强度大和基因组数据远超表型标签的问题。

### 背景

流感A病毒抗原性变异速度快，需要频繁更新疫苗，但用于量化抗原性的血凝抑制检测劳动强度大且难以扩展，导致基因组数据远超可用表型标签，限制了传统监督模型的有效性。

### 目的

验证将预训练蛋白质语言模型(PLMs)与半监督学习(SSL)结合，能否在标记数据稀少的情况下保持高预测准确性。

### 方法

评估了自训练和标签传播两种SSL策略，使用ESM-2、ProtVec、ProtT5和ProtBert四种PLM衍生的嵌入应用于HA序列，通过嵌套交叉验证模拟不同标签可用率(25%、50%、75%和100%)，在四种IAV亚型(H1N1、H3N2、H5N1、H9N2)上测试。

### 主要发现

SSL在标签稀缺情况下始终提高了性能；ProtVec自训练产生最大相对增益；ESM-2高度稳健，仅用25%标记数据F1分数超过0.82；H1N1和H9N2预测准确率高，高度可变的H3N2具挑战性但SSL减轻了性能下降。

### 结论

将PLMs与SSL相结合可解决抗原性标记瓶颈，更有效利用未标记监测序列，支持快速变异优先排序和及时疫苗株选择。

### 翻译

流感A病毒(IAVs)以需要频繁疫苗更新的速度进行抗原性变异，然而用于量化抗原性的血凝抑制(HI)检测劳动强度大且难以扩展。因此，基因组数据远超可用表型标签，限制了传统监督模型的有效性。我们假设将预训练蛋白质语言模型(PLMs)与半监督学习(SSL)结合，可以在标记数据稀少的情况下保持高预测准确性。我们使用四种PLM衍生的嵌入(ESM-2、ProtVec、ProtT5、ProtBert)应用于血凝素(HA)序列，评估了自训练和标签传播两种SSL策略与完全监督基线的对比。嵌套交叉验证框架模拟了低标签环境(25%、50%、75%和100%标签可用性)，涵盖四种IAV亚型(H1N1、H3N2、H5N1、H9N2)。SSL在标签稀缺情况下始终提高了性能。ProtVec自训练产生最大相对增益，表明SSL可以补偿较低分辨率的表示。ESM-2保持高度稳健，仅使用25%标记数据即可获得超过0.82的F1分数，表明其嵌入捕获了关键的抗原决定簇。虽然H1N1和H9N2预测准确率高，但高度可变的H3N2亚型仍然具有挑战性，尽管SSL减轻了性能下降。这些发现证明，将PLMs与SSL相结合可以解决抗原性标记瓶颈，并更有效地利用未标记的监测序列，支持快速变异优先排序和及时疫苗株选择。


### 论文摘要

Influenza A viruses (IAVs) evolve antigenically at a pace that requires frequent vaccine updates, yet the haemagglutination inhibition (HI) assays used to quantify antigenicity are labor-intensive and unscalable. As a result, genomic data vastly outpace available phenotypic labels, limiting the effectiveness of traditional supervised models. We hypothesize that combining pre-trained Protein Language Models (PLMs) with Semi-Supervised Learning (SSL) can retain high predictive accuracy even when labeled data are scarce. We evaluated two SSL strategies, Self-training and Label Spreading, against fully supervised baselines using four PLM-derived embeddings (ESM-2, ProtVec, ProtT5, ProtBert) applied to haemagglutinin (HA) sequences. A nested cross-validation framework simulated low-label regimes (25%, 50%, 75%, and 100% label availability) across four IAV subtypes (H1N1, H3N2, H5N1, H9N2). SSL consistently improved performance under label scarcity. Self-training with ProtVec produced the largest relative gains, showing that SSL can compensate for lower-resolution representations. ESM-2 remained highly robust, achieving F1 scores above 0.82 with only 25% labeled data, indicating that its embeddings capture key antigenic determinants. While H1N1 and H9N2 were predicted with high accuracy, the hypervariable H3N2 subtype remained challenging, although SSL mitigated the performance decline. These findings demonstrate that integrating PLMs with SSL can address the antigenicity labeling bottleneck and enable more effective use of unlabeled surveillance sequences, supporting rapid variant prioritization and timely vaccine strain selection.

---

## 24. Task-Specific Trust Evaluation for Multi-Hop Collaborator Selection via GNN-Aided Distributed Agentic AI

**论文链接:** [http://arxiv.org/abs/2512.05788v1](http://arxiv.org/abs/2512.05788v1)

**作者:** Botao Zhu, Xianbin Wang, Dusit Niyato

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种名为GADAI的图神经网络辅助的分布式智能体AI框架，用于解决网络设备间协作任务中可信协作者选择的问题。该框架通过分别评估并整合设备任务特定可信度的不同方面，实现了更准确的多跳协作者选择。

### 背景

网络设备间协作任务的成功完成依赖于对可信协作者的有效选择。然而，对多跳协作者的准确任务特定信任评估极其复杂，因为它取决于多种具有不同特性的信任相关视角的组合，包括历史协作可靠性、可用资源的波动性和敏感条件，以及不断演变的网络拓扑结构。

### 目的

解决多跳协作者任务特定信任评估的复杂性，提出一个能够分别评估并联合整合设备任务特定可信度不同方面的框架，以促进多跳协作者选择。

### 方法

GADAI框架包含两个主要组件：1) 使用GNN辅助模型从历史协作数据推断设备信任，利用GNN传播和聚合多跳邻居间的信任信息；2) 实现隐私保护资源评估机制，每个设备托管由大型AI模型驱动的智能体，能够自主确定本地资源是否满足任务要求。

### 主要发现

通过结合信任评估和资源评估的结果，只有可信设备能通过其智能体以分布式方式协调面向任务的多跳合作路径。实验表明，GADAI在规划最大化任务完成价值的多跳路径方面优于比较算法。

### 结论

GADAI框架通过结合图神经网络和智能体AI，有效解决了网络设备间协作任务中多跳协作者信任评估的复杂性问题，实现了更准确的多跳路径规划和更高的任务完成价值。

### 翻译

网络设备间协作任务的成功完成取决于对可信协作者的有效选择。然而，对多跳协作者的准确任务特定信任评估可能极其复杂。原因是他们的信任评估取决于具有不同特性的多样化信任相关视角的组合，包括历史协作可靠性、可用资源的波动性和敏感条件，以及不断演变的网络拓扑结构。为了应对这一挑战，本文提出了一个图神经网络辅助的分布式智能体AI（GADAI）框架，在该框架中，设备的任务特定可信度的不同方面被分别评估并联合整合，以促进多跳协作者选择。GADAI首先利用GNN辅助模型从历史协作数据推断设备信任。具体而言，它使用GNN传播和聚合多跳邻居之间的信任信息，从而实现更准确的设备可靠性评估。考虑到设备资源的动态性和隐私敏感性，使用智能体AI实现了隐私保护资源评估机制。每个设备托管一个由大型AI模型驱动的智能体，能够自主确定其本地资源是否满足给定任务的要求，确保任务特定和隐私保护的信任评估。通过结合这些评估的结果，只有可信设备才能通过其智能体以分布式方式协调面向任务的多跳合作路径。实验结果表明，我们提出的GADAI在规划最大化任务完成价值的多跳路径方面优于比较算法。


### 论文摘要

The success of collaborative task completion among networked devices hinges on the effective selection of trustworthy collaborators. However, accurate task-specific trust evaluation of multi-hop collaborators can be extremely complex. The reason is that their trust evaluation is determined by a combination of diverse trust-related perspectives with different characteristics, including historical collaboration reliability, volatile and sensitive conditions of available resources for collaboration, as well as continuously evolving network topologies. To address this challenge, this paper presents a graph neural network (GNN)-aided distributed agentic AI (GADAI) framework, in which different aspects of devices' task-specific trustworthiness are separately evaluated and jointly integrated to facilitate multi-hop collaborator selection. GADAI first utilizes a GNN-assisted model to infer device trust from historical collaboration data. Specifically, it employs GNN to propagate and aggregate trust information among multi-hop neighbours, resulting in more accurate device reliability evaluation. Considering the dynamic and privacy-sensitive nature of device resources, a privacy-preserving resource evaluation mechanism is implemented using agentic AI. Each device hosts a large AI model-driven agent capable of autonomously determining whether its local resources meet the requirements of a given task, ensuring both task-specific and privacy-preserving trust evaluation. By combining the outcomes of these assessments, only the trusted devices can coordinate a task-oriented multi-hop cooperation path through their agents in a distributed manner. Experimental results show that our proposed GADAI outperforms the comparison algorithms in planning multi-hop paths that maximize the value of task completion.

---

## 25. 论文ID: 2512.05764v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.05764v1.json'

---

## 26. 论文ID: 2512.05683v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.05683v1.json'

---

## 27. Bounded Graph Clustering with Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.05623v1](http://arxiv.org/abs/2512.05623v1)

**作者:** Kibidi Neocosmos, Diego Baptista, Nicole Ludwig

**发布时间:** 2025-12-05

**备注:** 17 pages, 8 figures

### GPT解析

### 总结

本研究提出了一种灵活且原则性的方法来控制图神经网络（GNNs）在社区检测中发现的社区数量，解决了传统GNN方法无法精确返回指定聚类数量的问题。

### 背景

在社区检测领域，许多方法需要用户预先指定聚类数量，因为对所有可能值进行穷举搜索在计算上不可行。虽然一些经典算法可以直接从数据中推断出这个数量，但对于图神经网络通常不是这样。即使指定了期望的聚类数量，标准的基于GNN的方法也常常无法返回精确的数量，这是由它们的设计方式决定的。

### 目的

本研究旨在解决图神经网络在社区检测中无法精确返回指定聚类数量的局限性，提供一种灵活且原则性的方法来控制GNN发现的社区数量。

### 方法

研究团队提出了一种框架，该框架不假设真实聚类数量是已知的，允许用户指定一个合理的范围并在训练过程中强制执行这些边界。如果用户想要精确数量的聚类，也可以指定并可靠地返回。

### 主要发现

通过提出的框架，用户可以灵活地控制社区检测的聚类数量，既可以指定一个合理的范围，也可以指定精确的数量，并且能够可靠地获得期望的结果。

### 结论

这项工作通过引入一种灵活且原则性的方法，成功解决了图神经网络在社区检测中无法精确返回指定聚类数量的局限性，为社区检测领域提供了新的解决方案。

### 翻译

在社区检测中，许多方法需要用户预先指定聚类数量，因为对所有可能值进行穷举搜索在计算上不可行。虽然一些经典算法可以直接从数据中推断出这个数量，但对于图神经网络通常不是这样：即使当指定了期望的聚类数量时，标准的基于GNN的方法也常常由于它们的设计方式而无法返回精确的数量。在这项工作中，我们通过引入一种灵活且原则性的方法来控制GNN发现的社区数量，解决了这一局限性。我们不假设真实聚类数量是已知的，而是提出一个框架，允许用户指定一个合理的范围并在训练过程中强制执行这些边界。然而，如果用户想要精确数量的聚类，也可以指定并可靠地返回。


### 论文摘要

In community detection, many methods require the user to specify the number of clusters in advance since an exhaustive search over all possible values is computationally infeasible. While some classical algorithms can infer this number directly from the data, this is typically not the case for graph neural networks (GNNs): even when a desired number of clusters is specified, standard GNN-based methods often fail to return the exact number due to the way they are designed. In this work, we address this limitation by introducing a flexible and principled way to control the number of communities discovered by GNNs. Rather than assuming the true number of clusters is known, we propose a framework that allows the user to specify a plausible range and enforce these bounds during training. However, if the user wants an exact number of clusters, it may also be specified and reliably returned.

---

## 28. 论文ID: 2512.05475v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.05475v1.json'

---

## 29. Beyond Detection: A Comprehensive Benchmark and Study on Representation Learning for Fine-Grained Webshell Family Classification

**论文链接:** [http://arxiv.org/abs/2512.05288v1](http://arxiv.org/abs/2512.05288v1)

**作者:** Feijiang Han

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了一种自动化WebShell家族分类的方法，通过提取动态函数调用轨迹并利用大型语言模型增强数据集，结合多种表示方法进行评估，建立了该领域的强大基线。

### 背景

恶意WebShell对关键数字基础设施构成重大威胁，特别是在医疗和金融等公共服务领域。尽管研究社区在WebShell检测方面已取得进展，但当前仍主要依赖被动检测。

### 目的

从被动检测转向深入分析和主动防御，通过自动化WebShell家族分类来识别特定恶意软件谱系，了解攻击者战术并实现精确快速响应。

### 方法

提取动态函数调用轨迹捕获固有行为，使用大型语言模型合成新变体增强数据集，将轨迹抽象为序列、图和树，并评估多种表示方法（序列嵌入、transformers和结构感知算法）。

### 主要发现

通过在四个真实世界、家族注释的数据集上进行实验，建立了强大的基线，并确定了最有效的数据抽象、表示模型和学习范式组合。

### 结论

WebShell家族分类是一个 largely unexplored 的领域，本文提出的自动化方法为该领域提供了重要进展，有助于实现更有效的防御策略。

### 翻译

恶意WebShell通过破坏关键数字基础设施并对医疗和金融等领域的公共服务构成威胁，构成了重大且不断演变的威胁。尽管研究社区在WebShell检测方面取得了显著进展（即区分恶意样本和良性样本），我们认为现在是时候从被动检测转向深入分析和主动防御了。一个有希望的方向是WebShell家族分类的自动化，这涉及识别特定的恶意软件谱系，以便了解攻击者的战术并实现精确、快速的响应。然而，这一关键任务在很大程度上仍是一个未被充分探索的领域，目前依赖缓慢的手动专家分析。为了解决这一差距，我们提出了首个系统性研究来自动化WebShell家族分类。我们的方法首先提取动态函数调用轨迹，以捕获对常见加密和混淆具有抵抗性的固有行为。为了增强数据集的规模和多样性以获得更稳定的评估，我们通过大型语言模型合成的新变体来增强这些真实世界的轨迹。然后将这些增强的轨迹抽象为序列、图和树，为基准测试全面的表示方法套件奠定基础。我们在四个真实世界、家族注释的数据集上进行了广泛的实验，涵盖监督和非监督设置，建立了强大的基线，并为这一挑战中最有效的数据抽象、表示模型和学习范式组合提供了实用见解。


### 论文摘要

Malicious WebShells pose a significant and evolving threat by compromising critical digital infrastructures and endangering public services in sectors such as healthcare and finance. While the research community has made significant progress in WebShell detection (i.e., distinguishing malicious samples from benign ones), we argue that it is time to transition from passive detection to in-depth analysis and proactive defense. One promising direction is the automation of WebShell family classification, which involves identifying the specific malware lineage in order to understand an adversary's tactics and enable a precise, rapid response. This crucial task, however, remains a largely unexplored area that currently relies on slow, manual expert analysis. To address this gap, we present the first systematic study to automate WebShell family classification. Our method begins with extracting dynamic function call traces to capture inherent behaviors that are resistant to common encryption and obfuscation. To enhance the scale and diversity of our dataset for a more stable evaluation, we augment these real-world traces with new variants synthesized by Large Language Models. These augmented traces are then abstracted into sequences, graphs, and trees, providing a foundation to benchmark a comprehensive suite of representation methods. Our evaluation spans classic sequence-based embeddings (CBOW, GloVe), transformers (BERT, SimCSE), and a range of structure-aware algorithms, including Graph Kernels, Graph Edit Distance, Graph2Vec, and various Graph Neural Networks. Through extensive experiments on four real-world, family-annotated datasets under both supervised and unsupervised settings, we establish a robust baseline and provide practical insights into the most effective combinations of data abstractions, representation models, and learning paradigms for this challenge.

---

## 30. 论文ID: 2512.05287v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.05287v1.json'

---

## 31. Edged Weisfeiler-Lehman Algorithm

**论文链接:** [http://arxiv.org/abs/2512.05238v1](http://arxiv.org/abs/2512.05238v1)

**作者:** Xiao Yue, Bo Liu, Feng Zhang, Guangzhi Qu

**发布时间:** 2025-12-04

**DOI:** 10.1007/978-3-031-72344-5_7

**备注:** Author's Accepted Manuscript (AAM) of ICANN 2024 paper published in LNCS (Springer). Final version available at: https://link.springer.com/chapter/10.1007/978-3-031-72344-5_7

### GPT解析

### 总结

本文提出了Edged-WL算法(E-WL)和Edged Graph Isomorphism Network(EGIN)模型，扩展了传统的Weisfeiler-Lehman算法以利用边特征，解决了许多图神经网络不利用边特征的问题。实验表明，在图分类任务中，EGIN模型表现出优越的性能。

### 背景

传播-聚合方法是图学习的经典方法，被许多图神经网络使用，通过递归聚合节点自身和邻居节点的表示来更新节点表示。Weisfeiler-Lehman(1-WL)算法通过节点及其邻居节点的颜色表示测试同构性，但该算法不利用任何边特征，存在改进空间。

### 目的

解决1-WL算法不利用边特征的局限性，开发一种能够有效利用边特征的图学习算法和模型，以提高图学习性能。

### 方法

提出了Edged-WL算法(E-WL)，扩展了原始1-WL算法以整合边特征；基于E-WL算法，引入了Edged Graph Isomorphism Network(EGIN)模型，进一步利用边特征，解决了许多GNN不利用图数据边特征的问题。

### 主要发现

在使用12个具有边特征的基准图数据集进行的实验中，提出的EGIN模型与最先进的基线模型相比，在图分类任务中表现出优越的性能。

### 结论

E-WL算法和EGIN模型能够有效利用边特征，提高图学习性能，特别是在图分类任务中优于其他最先进的方法。

### 翻译

作为图学习的经典方法，传播-聚合方法论被许多图神经网络广泛采用，其中节点的表示通过递归聚合自身和邻居节点的表示来更新。类似于传播-聚合方法，Weisfeiler-Lehman(1-WL)算法根据节点及其邻居节点的颜色表示通过颜色细化测试同构性。然而，1-WL不利用任何边特征（标签），在某些领域利用边特征存在潜在改进空间。为解决这一局限性，我们提出了新颖的Edged-WL算法（E-WL），它扩展了原始1-WL算法以整合边特征。基于E-WL算法，我们还引入了Edged Graph Isomorphism Network（EGIN）模型，以进一步利用边特征，解决了许多GNN不利用图数据任何边特征的一个关键缺点。我们使用12个具有边特征的基准图数据集评估了所提出模型的性能，并与一些最先进的基线模型进行了比较。实验结果表明，我们提出的EGIN模型在图分类任务的图学习中通常表现出优越的性能。


### 论文摘要

As a classical approach on graph learning, the propagation-aggregation methodology is widely exploited by many of Graph Neural Networks (GNNs), wherein the representation of a node is updated by aggregating representations from itself and neighbor nodes recursively. Similar to the propagation-aggregation methodology, the Weisfeiler-Lehman (1-WL) algorithm tests isomorphism through color refinement according to color representations of a node and its neighbor nodes. However, 1-WL does not leverage any edge features (labels), presenting a potential improvement on exploiting edge features in some fields. To address this limitation, we proposed a novel Edged-WL algorithm (E-WL) which extends the original 1-WL algorithm to incorporate edge features. Building upon the E-WL algorithm, we also introduce an Edged Graph Isomorphism Network (EGIN) model for further exploiting edge features, which addresses one key drawback in many GNNs that do not utilize any edge features of graph data. We evaluated the performance of proposed models using 12 edge-featured benchmark graph datasets and compared them with some state-of-the-art baseline models. Experimental results indicate that our proposed EGIN models, in general, demonstrate superior performance in graph learning on graph classification tasks.

---

## 32. QoSDiff: An Implicit Topological Embedding Learning Framework Leveraging Denoising Diffusion and Adversarial Attention for Robust QoS Prediction

**论文链接:** [http://arxiv.org/abs/2512.04596v2](http://arxiv.org/abs/2512.04596v2)

**作者:** Guanchen Du, Jianlong Xu, Wei Wei

**发布时间:** 2025-12-04

**备注:** Preprint submitted to IEEE Transactions on Services Computing

### GPT解析

### 总结

本文提出了QoSDiff，一种新的嵌入学习框架，用于服务质量预测，绕过了传统方法中显式图构建的需求，通过去噪扩散概率模型和对抗交互模块有效捕获用户-服务间的复杂关系，在大规模数据集上表现出优越的性能和鲁棒性。

### 背景

准确的服务质量(QoS)预测对服务计算至关重要，能为服务选择提供数据驱动的指导并确保优质用户体验。然而，现有方法（尤其是图神经网络GNNs）严重依赖构建显式的用户-服务交互图，这在大规模场景下难以处理，且难以建模隐式拓扑关系，容易受环境噪声和异常值影响。

### 目的

解决传统QoS预测方法中显式图构建的局限性，开发一种无需显式图构建的嵌入学习框架，能够有效捕获用户-服务间的复杂关系，提高预测准确性和鲁棒性。

### 方法

提出QoSDiff框架，包含两个关键组件：1) 使用去噪扩散概率模型从噪声初始化中恢复内在的潜在结构；2) 设计对抗交互模块，整合双向混合注意力机制，捕获高阶交互，通过对抗范式动态区分信息模式与噪声，实现复杂用户-服务关联的双视角建模。

### 主要发现

在两个大规模真实数据集上的实验表明，QoSDiff显著优于最先进的基线方法。特别值得注意的是，该框架展现出卓越的跨数据集泛化能力和对观测噪声的出色鲁棒性。

### 结论

QoSDiff通过绕过显式图构建的需求，有效解决了传统QoS预测方法在大规模场景下的局限性，为服务计算中的服务质量预测提供了新的有效解决方案。

### 翻译

准确的服务质量(QoS)预测是服务计算的基础，为服务选择提供必要的数据驱动指导，确保优质用户体验。然而，现有方法（尤其是图神经网络GNNs）严重依赖构建显式的用户-服务交互图。这种依赖不仅导致大规模场景下显式图构建的不可行性，还限制了隐式拓扑关系的建模，并加剧了对环境噪声和异常值的敏感性。为解决这些挑战，本文引入QoSDiff，一种新颖的嵌入学习框架，绕过了显式图构建的先决条件。具体而言，它利用去噪扩散概率模型从噪声初始化中恢复内在的潜在结构。为进一步捕获高阶交互，我们提出了一个对抗交互模块，整合了双向混合注意力机制。这种对抗范式动态区分信息模式与噪声，实现了复杂用户-服务关联的双视角建模。在两个大规模真实数据集上的广泛实验表明，QoSDiff显著优于最先进的基线方法。值得注意的是，结果突显了该框架卓越的跨数据集泛化能力和对观测噪声的出色鲁棒性。


### 论文摘要

Accurate Quality of Service (QoS) prediction is fundamental to service computing, providing essential data-driven guidance for service selection and ensuring superior user experiences. However, prevalent approaches, particularly Graph Neural Networks (GNNs), heavily rely on constructing explicit user--service interaction graphs. Such reliance not only leads to the intractability of explicit graph construction in large-scale scenarios but also limits the modeling of implicit topological relationships and exacerbates susceptibility to environmental noise and outliers. To address these challenges, this paper introduces \emph{QoSDiff}, a novel embedding learning framework that bypasses the prerequisite of explicit graph construction. Specifically, it leverages a denoising diffusion probabilistic model to recover intrinsic latent structures from noisy initializations. To further capture high-order interactions, we propose an adversarial interaction module that integrates a bidirectional hybrid attention mechanism. This adversarial paradigm dynamically distinguishes informative patterns from noise, enabling a dual-perspective modeling of intricate user--service associations. Extensive experiments on two large-scale real-world datasets demonstrate that QoSDiff significantly outperforms state-of-the-art baselines. Notably, the results highlight the framework's superior cross-dataset generalization capability and exceptional robustness against observational noise.

---

## 33. Probing the effectiveness of World Models for Spatial Reasoning through Test-time Scaling

**论文链接:** [http://arxiv.org/abs/2512.05809v1](http://arxiv.org/abs/2512.05809v1)

**作者:** Saurav Jha, M. Jehanzeb Mirza, Wei Lin, Shiqi Yang, Sarath Chandar

**发布时间:** 2025-12-05

**备注:** Extended abstract at World Modeling Workshop 2026

### GPT解析

### 总结

本研究系统评估了测试时验证器在视觉语言模型空间推理任务中的表现，并提出了改进框架ViSA，揭示了当前世界模型在细粒度推理方面的局限性。

### 背景

视觉语言模型在需要多视图理解和身体视角转换的空间推理任务中存在局限性，近期方法如MindJourney通过测试时扩展试图弥补这一差距。

### 目的

系统研究测试时验证器在各种基准测试中的行为，揭示其潜力和缺陷，并提出改进方法。

### 方法

进行基于不确定性的分析，引入Verification through Spatial Assertions (ViSA)框架，将测试时奖励建立在可验证的、以帧为锚点的微观主张上。

### 主要发现

MindJourney验证器缺乏有效校准，随机评分效果相当；ViSA框架在SAT-Real基准上改进空间推理并纠正轨迹选择偏差；但在MMSI-Bench上所有验证器无法实现一致扩展，当前世界模型形成信息瓶颈。

### 结论

这些发现共同绘制了基于世界模型的推理中测试时验证的好、坏和丑陋方面，指出了未来研究方向。

### 翻译

视觉语言模型在需要多视图理解和身体视角转换的空间推理任务中仍然存在局限性。最近的方法如MindJourney试图通过测试时扩展来弥补这一差距，其中世界模型想象基于动作的条件轨迹，启发式验证器从这些轨迹中选择有用的视图。在这项工作中，我们系统研究了这些测试时验证器在各种基准测试中的行为，揭示了它们的潜力和缺陷。我们的基于不确定性的分析表明，MindJourney的验证器几乎没有提供有意义的校准，随机评分通常同样能减少答案熵，从而暴露了系统性的动作偏差和不可靠的奖励信号。为了缓解这些问题，我们引入了一种通过空间断言进行验证(ViSA)的框架，该框架将测试时奖励建立在可验证的、以帧为锚点的微观主张上。这种原则性的验证器在SAT-Real基准测试上持续改进空间推理，并通过更加平衡的探索性行为纠正了轨迹选择偏差。然而，在具有挑战性的MMSI-Bench上，包括我们提出的验证器在内的所有验证器都无法实现一致的扩展，这表明当前的世界模型形成了信息瓶颈，其中想象中的视图无法丰富细粒度推理。总的来说，这些发现共同绘制了基于世界模型的推理中测试时验证的好、坏和丑陋方面。我们的代码可在https://github.com/chandar-lab/visa-for-mindjourney获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉-语言模型（VLMs）在空间推理任务中的局限性问题，特别是在需要多视角理解和具身视角转换的复杂空间推理方面。这个问题在研究中很重要，因为空间推理是AI的基本能力，涉及推断3D结构、物体关系和跨视角转换；在现实中，这种能力对机器人导航、增强现实、自动驾驶等应用至关重要，提升VLMs的空间推理能力可以扩展它们在现实世界中的应用范围。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有方法MindJourney的局限性，发现其启发式验证器提供的有意义的校准很少，且往往强化系统性的动作偏差。通过不确定性分析，作者发现随机选择有时比MindJourney的评分更能减少答案熵。作者借鉴了'proposer-solver paradigm'（提议者-求解者范式），但创新性地将其应用于测试时间计算扩展，而不是仅用于训练时间的自我改进。作者还参考了世界模型的概念，使用预训练的视频扩散世界模型作为马尔可夫决策过程。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是提出'Verification through Spatial Assertions'（ViSA）框架，将测试时间验证分解为两个可解释的步骤：主张生成和主张验证。这种方法将评估想象帧的过程分解为：1）主张生成阶段：为每个候选帧生成描述空间关系、物体属性或动态变化的微主张；2）主张验证阶段：验证每个主张，产生判决（支持、矛盾或信息不足）和置信度分数；3）证据质量评分：基于验证结果，为每个想象帧分配证据质量（EQ）分数作为测试时间奖励；4）选择过程：使用EQ分数选择最有用的帧用于回答空间推理问题。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）ViSA框架：第一个将提议者-求解者交互应用于测试时间计算扩展的技术；2）微主张验证：将评估过程分解为细粒度的、可验证的空间主张；3）证据质量评分：结合主张覆盖率和验证器置信度；4）系统性行为分析：首次系统分析测试时间验证器在不同基准上的行为。相比之前的工作，ViSA基于可验证的空间主张而非启发式评分进行评估；应用于测试时间而非训练时间；能纠正轨迹选择偏差，产生更平衡的探索行为；在SAT-Real上表现优于随机选择，但在更复杂的MMSI-Bench上面临信息瓶颈。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过提出基于空间主张的验证框架ViSA，改进了世界模型在空间推理任务中的测试时间扩展效果，同时揭示了当前世界模型在细粒度推理任务中的信息瓶颈限制。'}


### 论文摘要

Vision-Language Models (VLMs) remain limited in spatial reasoning tasks that require multi-view understanding and embodied perspective shifts. Recent approaches such as MindJourney attempt to mitigate this gap through test-time scaling where a world model imagines action-conditioned trajectories and a heuristic verifier selects helpful views from such trajectories. In this work, we systematically examine how such test-time verifiers behave across benchmarks, uncovering both their promise and their pitfalls. Our uncertainty-based analyses show that MindJourney's verifier provides little meaningful calibration, and that random scoring often reduces answer entropy equally well, thus exposing systematic action biases and unreliable reward signals. To mitigate these, we introduce a Verification through Spatial Assertions (ViSA) framework that grounds the test-time reward in verifiable, frame-anchored micro-claims. This principled verifier consistently improves spatial reasoning on the SAT-Real benchmark and corrects trajectory-selection biases through more balanced exploratory behavior. However, on the challenging MMSI-Bench, none of the verifiers, including ours, achieve consistent scaling, suggesting that the current world models form an information bottleneck where imagined views fail to enrich fine-grained reasoning. Together, these findings chart the bad, good, and ugly aspects of test-time verification for world-model-based reasoning. Our code is available at https://github.com/chandar-lab/visa-for-mindjourney.

---

## 34. Symmetry-driven phonon confinement in 2D halide perovskites

**论文链接:** [http://arxiv.org/abs/2512.05792v1](http://arxiv.org/abs/2512.05792v1)

**作者:** Mustafa Mahmoud Aboulsaad, Olivier Donzel-Gargand, Rafael B. Araujo, Tomas Edvinsson

**发布时间:** 2025-12-05

**备注:** 22 pages, 4 figures

### GPT解析

### 总结

该研究探讨了量子限制对卤化物钙钛矿振动模式对称性的影响，通过合成不同厚度的2D CsPbBr3纳米片，揭示了振动模式的对称性差异及其与结构的关系，并建立了基于拉曼光谱的无损厚度测量方法。

### 背景

量子限制不仅重塑电子态，还重组了低维半导体的振动景观。然而，在卤化物钙钛矿中，限制在控制对称性对振动模式影响方面的作用仍未解决。

### 目的

研究2D CsPbBr3纳米片的量子限制效应对振动模式对称性的影响，探索对称工程作为理解和控制低维光电子材料中声子、激子及其相互作用的途径。

### 方法

合成具有原子级定义厚度的2-5单层(MLs)的2D CsPbBr3纳米片，进行激子吸收和发射分析、晶相确定和声子分析，使用偏振分辨拉曼光谱和第一性原理理论研究振动模式的对称性特征。

### 主要发现

最低维结构(2 MLs)显示立方和正交结构共存，3 MLs及以上能量上收敛为正交结构；B1g模式增强并遵循声子限制模型，而Ag模式偏离，反映不同空间局域化；B1g振动主要位于xy平面，A1g振动与面外畸变耦合且对表面无序敏感；建立了基于A1g/B1g强度比的拉曼指纹作为2D纳米片厚度的无损测量方法。

### 结论

这些结果连接了电子和声子限制，突出了对称工程作为理解和控制低维光电子材料中声子、激子及其相互作用的途径。

### 翻译

量子限制不仅重塑电子态，还重组了低维半导体的振动景观。然而，在卤化物钙钛矿中，限制在控制对称性对振动模式影响方面的作用仍未解决。在此，我们合成了具有原子级定义厚度的2-5单层(MLs)的2D CsPbBr3纳米片，并进行激子吸收和发射分析、晶相确定和声子分析。最低维结构(2 MLs)显示立方和正交结构共存，3 MLs及以上能量上收敛为正交结构。通过偏振分辨拉曼光谱和2-5 MLs的第一性原理理论，发现显著的对称性对比：B1g模式增强并遵循声子限制模型，而Ag模式偏离，反映了它们不同的空间局域化。第一性原理计算显示B1g振动模式主要位于xy平面，Pb-Br-Pb单元沿xy方向连接八面体，随着内层积累，晶格动力学增强；而A1g振动与面外畸变耦合，且对表面无序和有限尺寸效应保持敏感。这种对称性驱动的二分法为理解层状卤化物钙钛矿中的声子局域化提供了通用框架。除了机制，我们建立了拉曼指纹，特别是交叉偏振几何中A1g/B1g强度比，作为2-5 MLs范围内2D纳米片厚度的校准、无损测量方法。这些结果连接了电子和声子限制，突出了对称工程作为理解和控制低维光电子材料中声子、激子及其相互作用的途径。


### 论文摘要

Quantum confinement not only reshapes electronic states but also reorganizes the vibrational landscape of low-dimensional semiconductors. In halide perovskites, however, the role of confinement in governing symmetry effects on vibrational modes has remained unresolved. Here we synthesize 2D CsPbBr3 nanoplatelets with atomically defined thicknesses for 2-5 monolayers (MLs) and perform exciton absorption and emission analysis, crystalline phase determination, and phonon analysis. The lowest dimensional structure (2 MLs) reveal a co-existence of cubic and orthorhombic structure, energetically converging to orthorhombic for 3 MLs and beyond. Through polarization-resolved Raman spectroscopy and first-principles theory for 2-5 MLs, a striking symmetry contrast is found: B1g modes intensify and evolve in line with the phonon-confinement model, while Ag modes deviate, reflecting their distinct spatial localization. First principles calculations show that B1g vibrational modes largely reside in the xy-plane, Pb-Br-Pb units connect octahedra along the xy-direction with increased lattice dynamics as inner layers accumulate, whereas A1g vibrations couple to out-of-plane distortions and remain susceptible to surface disorder and finite-size effects. This symmetry-driven dichotomy provides a general framework for understanding phonon localization in layered halide perovskites. Beyond mechanism, we establish Raman fingerprints, particularly the A1g/B1g intensity ratio in cross-polarized geometry, as a calibrated, non-destructive metrology for 2D nanoplatelet thickness through 2-5 MLs. These results bridge electronic and phonon confinement and highlight symmetry engineering as a route to understand and control phonons, excitons, and their interactions in low-dimensional optoelectronic materials.

---

## 35. Nature of continuous spectra in wall-bounded shearing flows of FENE-P fluids

**论文链接:** [http://arxiv.org/abs/2512.05787v1](http://arxiv.org/abs/2512.05787v1)

**作者:** Pratyush Kumar Mohanty, P. S. D. Surya Phani Tej, Ganesh Subramanian, V. Shankar

**发布时间:** 2025-12-05

**备注:** 30 pages, 25 figures

### GPT解析

### 总结

本研究分析了FENE-P流体剪切流中的连续谱(CS)特性，发现与Oldroyd-B流体相比，FENE-P流体最多有六种不同的连续谱，其中三个与溶剂-溶液粘度比无关，另外三个与该参数相关，其中两个是FENE-P谱的新特征。

### 背景

用于模拟聚合物应力的本构方程在空间上是局部性的，导致有界粘弹性剪切流的线性化动力学微分算子存在奇点。这使得这类剪切流的特征谱除了离散特征值外，还包含由奇异特征函数组成的连续谱。理解理论CS轨迹对区分物理真实的离散特征值和 poorly approximated 的数值CS至关重要。

### 目的

提供对FENE-P流体直线和曲线剪切流中连续谱(CS)性质的全面描述，并与Oldroyd-B流体的CS进行对比。

### 方法

通过解析分析方法研究FENE-P流体剪切流的连续谱特性。

### 主要发现

对于FENE-P流体的剪切流，最多存在六种不同的连续谱。当有限可延伸参数L > 50(适用于实验中使用的大分子量聚合物)时，其中三个CS几乎相同且与溶剂-溶液粘度比(β)无关；另外三个CS与β相关，其中一个是Oldroyd-B流体中溶剂连续谱的类似物；剩余的两个与β相关的CS是FENE-P谱的新特征，其相速度可能超出基态速度范围，包括负值。

### 结论

FENE-P流体剪切流中预测的连续谱复杂性可能会扩展到其他表现出剪切稀疏流变学的非线性粘弹性模型。

### 翻译

由于通常用于模拟聚合物应力的本构方程具有空间局部性，控制有界粘弹性剪切流线性化动力学的微分算子具有奇点。因此，这类剪切流的特征谱除了包含离散特征值外，还包含由奇异特征函数组成的连续谱(CS)。对理论CS轨迹的清晰理解对于区分物理真实的(离散)特征值和 poorly approximated 的数值CS至关重要。对于Oldroyd-B流体的直线剪切流，CS是一对线段，其长度等于基态速度范围。在本研究中，我们首次提供了对FENE-P流体直线和曲线剪切流中CS性质的全面描述。与上述Oldroyd-B流体的CS形成鲜明对比，我们通过解析分析表明，FENE-P流体剪切流最多有六种不同的连续谱。当有限可延伸参数L > 50(适用于实验中使用的大分子量聚合物)时，其中三个CS几乎相同，且与溶剂-溶液粘度比(β)无关。另外三个CS与β相关，其中一个是Oldroyd-B流体中溶剂(粘性)连续谱的类似物。剩余的两个与β相关的CS是FENE-P谱的新特征，其相速度可能超出基态速度范围，包括负值。本研究预测的FENE-P流体剪切流的CS复杂性预计会扩展到其他表现出剪切稀疏流变学的非线性粘弹性模型。


### 论文摘要

Owing to the spatially local nature of the constitutive equations typically used to model polymeric stresses, the differential operators governing the linearized dynamics of bounded viscoelastic shearing flows have singular points. As a result, the eigenspectra of such shearing flows contain, in addition to discrete eigenvalues, continuous spectra (CS) comprising singular eigenfunctions. A clear understanding of the theoretical CS loci is crucial in discriminating physically genuine (discrete) eigenvalues from the poorly approximated numerical CS. For rectilinear shear flows of Oldroyd-B fluids, the CS are a pair of line segments, with lengths equal to the base-state range of velocities. In this study, we provide the first comprehensive account of the nature of the CS for both rectilinear and curvilinear shearing flows of the FENE-P fluid. In stark contrast to the CS for the Oldroyd-B fluid mentioned above, we show analytically that there are up to six distinct continuous spectra for shearing flows of FENE-P fluids. When the finite extensibility parameter $L > 50$, as appropriate for large molecular weight polymers used in experiments, three of the CS are nearly identical, and independent of the solvent-to-solution viscosity ratio ($β$). The other three CS are $β$-dependent, with one of them being the analogue of the solvent (viscous) continuous spectrum in the Oldroyd-B fluid. The remaining two $β$-dependent CS are novel features of the FENE-P spectrum, and can have phase speeds outside the base range of velocities, including negative ones. The complexity of the CS predicted here for shearing flows of FENE-P fluids is expected to carry over to other nonlinear viscoelastic models that exhibit a shear-thinning rheology.

---

## 36. Active Video Perception: Iterative Evidence Seeking for Agentic Long Video Understanding

**论文链接:** [http://arxiv.org/abs/2512.05774v1](http://arxiv.org/abs/2512.05774v1)

**作者:** Ziyang Wang, Honglu Zhou, Shijie Wang, Junnan Li, Caiming Xiong, Silvio Savarese, Mohit Bansal, Michael S. Ryoo, Juan Carlos Niebles

**发布时间:** 2025-12-05

**备注:** Website: https://activevideoperception.github.io/

### GPT解析

### 总结

该研究提出了主动视频感知(AVP)框架，通过智能体主动决定观察内容、时机和位置，高效获取与查询相关的视频证据，显著提升了长视频理解的效率和准确性。

### 背景

长视频理解具有挑战性，因为回答问题依赖于埋藏在冗余内容中的稀疏线索。现有智能体框架依赖与查询无关的标注器，导致计算资源浪费和细粒度时空信息模糊。

### 目的

开发一种能够让智能体主动决定观察什么、何时观察、何地观察，并能持续评估当前观察是否足够回答查询的长视频理解框架。

### 方法

提出主动视频感知(AVP)框架，采用迭代式'计划-观察-反思'过程：规划者提出针对性视频交互，观察者执行交互提取带时间戳的证据，反思者评估证据充分性并决定是否继续观察。

### 主要发现

在五个长视频理解基准测试中，AVP实现了最佳性能，平均准确率比最佳智能体方法提高5.7%，同时仅需18.4%的推理时间和12.4%的输入标记。

### 结论

主动视频感知框架通过选择性获取相关信息，有效解决了长视频理解中的效率与准确性问题，为视频理解领域提供了新思路。

### 翻译

长视频理解具有挑战性，因为回答现实世界的问题往往依赖于埋藏在数小时冗余和不相关内容中的稀疏、时间上分散的线索。虽然智能体管道改善了视频推理能力，但主流框架依赖于与查询无关的标注器来感知视频信息，这导致在无关内容上浪费计算资源，并模糊了细粒度的时间和空间信息。受主动感知理论启发，我们认为长视频理解智能体应主动决定观察什么内容、何时观察以及在哪里观察，并持续评估当前观察是否足以回答查询。我们提出了'主动视频感知'(AVP)框架，这是一个寻求证据的框架，将视频视为交互式环境，直接从像素中获取紧凑的、与查询相关的证据。具体而言，AVP使用大型语言模型智能体运行迭代式的'计划-观察-反思'过程。在每个回合中，规划者提出有针对性的视频交互，观察者执行这些交互以提取带时间戳的证据，反思者评估证据对查询的充分性，然后决定是停止并给出答案还是继续进一步观察。在五个长视频理解基准测试中，AVP实现了最高的性能，并有显著提升。值得注意的是，与最佳智能体方法相比，AVP的平均准确率提高了5.7%，同时仅需要18.4%的推理时间和12.4%的输入标记。


### 论文摘要

Long video understanding (LVU) is challenging because answering real-world queries often depends on sparse, temporally dispersed cues buried in hours of mostly redundant and irrelevant content. While agentic pipelines improve video reasoning capabilities, prevailing frameworks rely on a query-agnostic captioner to perceive video information, which wastes computation on irrelevant content and blurs fine-grained temporal and spatial information. Motivated by active perception theory, we argue that LVU agents should actively decide what, when, and where to observe, and continuously assess whether the current observation is sufficient to answer the query. We present Active Video Perception (AVP), an evidence-seeking framework that treats the video as an interactive environment and acquires compact, queryrelevant evidence directly from pixels. Concretely, AVP runs an iterative plan-observe-reflect process with MLLM agents. In each round, a planner proposes targeted video interactions, an observer executes them to extract time-stamped evidence, and a reflector evaluates the sufficiency of the evidence for the query, either halting with an answer or triggering further observation. Across five LVU benchmarks, AVP achieves highest performance with significant improvements. Notably, AVP outperforms the best agentic method by 5.7% in average accuracy while only requires 18.4% inference time and 12.4% input tokens.

---

## 37. Distilling Expert Surgical Knowledge: How to train local surgical VLMs for anatomy explanation in Complete Mesocolic Excision

**论文链接:** [http://arxiv.org/abs/2512.05740v1](http://arxiv.org/abs/2512.05740v1)

**作者:** Lennart Maack, Julia-Kristin Graß, Lisa-Marie Toscha, Nathaniel Melling, Alexander Schlaefer

**发布时间:** 2025-12-05

### GPT解析

### 总结

该研究提出了一种隐私保护框架，将大型通用语言模型的知识提炼到高效、本地部署的视觉语言模型中，用于外科手术场景理解，特别是在完整结肠系膜切除术中的解剖标志识别和解释。

### 背景

当前视觉大语言模型在计算机辅助诊断和决策支持方面显示出巨大潜力，但在特定领域的外科手术场景理解方面存在不足，例如在完整结肠系膜切除术中识别和解释解剖标志。此外，需要在本地部署模型以避免患者数据泄露到诊所外的大型VLMs。

### 目的

开发一种数据高效且符合隐私要求的训练方法，用于训练针对外科手术领域优化的、可在本地部署的视觉语言模型，以增强外科手术场景的理解能力。

### 方法

提出一种隐私保护框架，通过在不使用敏感图像的情况下，仅使用文本上下文和二元分割掩码作为空间信息来提示教师语言模型，生成专家监督数据集。然后使用此数据集进行监督微调（SFT）和后续的直接偏好优化（DPO），训练本地部署的视觉语言模型。

### 主要发现

评估确认，使用生成的数据集微调视觉语言模型可以显著提高其在外科手术领域的知识水平，与基础视觉语言模型相比有大幅提升。

### 结论

这项工作验证了一种数据高效且符合隐私要求的训练方法，用于训练针对外科手术领域优化的、可在本地部署的视觉语言模型，以实现外科手术场景理解。

### 翻译

最近，视觉大语言模型在计算机辅助诊断和决策支持方面显示出巨大潜力。然而，当前的VLMs在特定领域的外科手术场景理解方面存在不足，例如在完整结肠系膜切除术中识别和解释解剖标志。此外，需要本地部署的模型以避免患者数据泄露到诊所外的大型VLMs。我们提出了一种隐私保护框架，将大型通用语言模型的知识提炼到高效、本地部署的VLM中。我们通过在不使用敏感图像的情况下，仅使用文本上下文和二元分割掩码作为空间信息来提示教师语言模型，生成了专家监督数据集。该数据集用于本地部署的VLM的监督微调（SFT）和后续的直接偏好优化（DPO）。我们的评估确认，使用生成的数据集微调VLM可以显著提高其在外科手术领域的知识水平，与基础VLM相比有大幅提升。总体而言，这项工作验证了一种数据高效且符合隐私要求的训练方法，用于训练针对外科手术领域优化的、可在本地部署的VLM，以实现外科手术场景理解。


### 论文摘要

Recently, Vision Large Language Models (VLMs) have demonstrated high potential in computer-aided diagnosis and decision-support. However, current VLMs show deficits in domain specific surgical scene understanding, such as identifying and explaining anatomical landmarks during Complete Mesocolic Excision. Additionally, there is a need for locally deployable models to avoid patient data leakage to large VLMs, hosted outside the clinic. We propose a privacy-preserving framework to distill knowledge from large, general-purpose LLMs into an efficient, local VLM. We generate an expert-supervised dataset by prompting a teacher LLM without sensitive images, using only textual context and binary segmentation masks for spatial information. This dataset is used for Supervised Fine-Tuning (SFT) and subsequent Direct Preference Optimization (DPO) of the locally deployable VLM. Our evaluation confirms that finetuning VLMs with our generated datasets increases surgical domain knowledge compared to its base VLM by a large margin. Overall, this work validates a data-efficient and privacy-conforming way to train a surgical domain optimized, locally deployable VLM for surgical scene understanding.

---

## 38. LeAD-M3D: Leveraging Asymmetric Distillation for Real-time Monocular 3D Detection

**论文链接:** [http://arxiv.org/abs/2512.05663v1](http://arxiv.org/abs/2512.05663v1)

**作者:** Johannes Meier, Jonathan Michel, Oussema Dhaouadi, Yung-Hsu Yang, Christoph Reich, Zuria Bauer, Stefan Roth, Marc Pollefeys, Jacques Kaiser, Daniel Cremers

**发布时间:** 2025-12-05

### GPT解析

### 总结

LeAD-M3D是一种单目3D检测器，通过三个关键组件实现最先进准确性和实时推理，无需额外模态。

### 背景

实时单目3D物体检测面临深度歧义、视角变化和3D推理高计算成本的挑战，现有方法要么依赖LiDAR或几何先验，要么牺牲效率换取准确性。

### 目的

开发一种单目3D检测器，实现高准确性和实时推理效率，无需额外模态如LiDAR。

### 方法

LeAD-M3D包含三个关键组件：1)非对称增强去噪蒸馏(A2D2)转移几何知识；2)3D感知一致性匹配(CM3D)改进预测分配；3)置信度门控3D推理(CGI3D)加速检测。

### 主要发现

LeAD-M3D在KITTI和Waymo上实现最先进准确性，在Rope3D上达到最佳汽车AP，比之前的高精度方法快3.6倍，证明单目3D检测可同时实现高保真度和实时效率。

### 结论

单目3D检测的高保真度和实时效率可以同时实现，无需LiDAR、立体视觉或几何假设。

### 翻译

实时单目3D物体检测由于严重的深度歧义、视角变化和3D推理的高计算成本而具有挑战性。现有方法要么依赖LiDAR或几何先验来补偿缺失的深度，要么牺牲效率以实现有竞争力的准确性。我们引入LeAD-M3D，一种单目3D检测器，无需额外模态即可实现最先进的准确性和实时推理。我们的方法由三个关键组件驱动。非对称增强去噪蒸馏(A2D2)通过质量和重要性加权的深度特征损失，将干净图像教师的几何知识转移到混合噪声学生模型，无需LiDAR监督即可实现更强的深度推理。3D感知一致性匹配(CM3D)通过将3D MGIoU集成到匹配分数中改进预测到真实值的分配，产生更稳定和精确的监督。最后，置信度门控3D推理(CGI3D)通过将昂贵的3D回归限制在最高置信度区域来加速检测。这些组件共同为单目3D检测设定了新的帕累托前沿：LeAD-M3D在KITTI和Waymo上实现了最先进的准确性，在Rope3D上实现了最佳报告的汽车AP，同时比之前的高精度方法快3.6倍。我们的结果表明，单目3D检测中的高保真度和实时效率可以同时实现 - 无需LiDAR、立体视觉或几何假设。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决单目实时3D物体检测的三大挑战：严重的深度模糊性、视角变化和3D推理的高计算成本。这个问题在现实中非常重要，因为单目3D检测在自动驾驶、机器人、医学和城市基础设施等领域有广泛应用，而现有方法要么依赖LiDAR等额外模态（这些在许多场景中不可用），要么牺牲效率来获得准确性，难以满足实时应用的需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了单目3D检测的挑战和现有方法的局限性，然后考虑使用知识蒸馏来平衡准确性和效率。注意到现有方法依赖LiDAR等特权数据创建教师-学生不对称性，作者创新性地提出通过混合增强(mixup)创建信息不对称，同时保持几何不变性。作者借鉴了YOLOv10的高效检测框架、知识蒸馏技术、mixup数据增强和MGIoU匹配方法，但对这些方法进行了创新性改进，以适应单目3D检测的特殊需求。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过三个关键组件实现高效准确的3D检测：1)A2D2利用混合增强创建信息不对称，通过去噪任务转移几何知识；2)CM3D结合2D和3D重叠度改进预测与真实值分配；3)CGI3D将昂贵的3D回归限制在高置信度区域加速推理。整体流程基于YOLOv10构建5种规模的模型变体，先训练最大模型作为教师，再通过A2D2训练学生模型，使用CM3D进行匹配，最后通过CGI3D实现高效推理，在KITTI、Waymo和Rope3D等基准上评估性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)A2D2-无需LiDAR的知识蒸馏方案，利用混合增强创建不对称性；2)CM3D-结合2D和3D重叠度的匹配策略；3)CGI3D-将3D回归限制在高置信度区域；4)提供5种不同规模的模型变体。相比之前工作，不同之处在于：无需额外模态创建不对称性；根据预测质量和特征重要性动态加权特征损失；结合2D和3D几何质量进行匹配；仅在高置信度区域进行推理；提供多种模型规模选择。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LeAD-M3D通过创新的无需LiDAR的知识蒸馏、3D感知匹配和置信度门控推理，实现了在保持最先进准确性的同时达到实时性能的单目3D物体检测，为自动驾驶和机器人应用提供了高效且准确的3D感知解决方案。'}


### 论文摘要

Real-time monocular 3D object detection remains challenging due to severe depth ambiguity, viewpoint shifts, and the high computational cost of 3D reasoning. Existing approaches either rely on LiDAR or geometric priors to compensate for missing depth, or sacrifice efficiency to achieve competitive accuracy. We introduce LeAD-M3D, a monocular 3D detector that achieves state-of-the-art accuracy and real-time inference without extra modalities. Our method is powered by three key components. Asymmetric Augmentation Denoising Distillation (A2D2) transfers geometric knowledge from a clean-image teacher to a mixup-noised student via a quality- and importance-weighted depth-feature loss, enabling stronger depth reasoning without LiDAR supervision. 3D-aware Consistent Matching (CM3D) improves prediction-to-ground truth assignment by integrating 3D MGIoU into the matching score, yielding more stable and precise supervision. Finally, Confidence-Gated 3D Inference (CGI3D) accelerates detection by restricting expensive 3D regression to top-confidence regions. Together, these components set a new Pareto frontier for monocular 3D detection: LeAD-M3D achieves state-of-the-art accuracy on KITTI and Waymo, and the best reported car AP on Rope3D, while running up to 3.6x faster than prior high-accuracy methods. Our results demonstrate that high fidelity and real-time efficiency in monocular 3D detection are simultaneously attainable - without LiDAR, stereo, or geometric assumptions.

---

## 39. EA-ERT: a new ensemble approach to convert time-lapse ERT data to soil water content

**论文链接:** [http://arxiv.org/abs/2512.05662v1](http://arxiv.org/abs/2512.05662v1)

**作者:** B. Loiseau, S. D. Carrière, N. K. Martin-StPaul, R. Clément, C. Champollion, V. Mercier, J. Thiesson, S. Pasquet, C. Doussan, T. Hermans, D. Jougnot

**发布时间:** 2025-12-05

### GPT解析

### 总结

本研究提出了一种创新的集合方法电阻率层析成像技术，通过构建电阻率集合模型并将其转换为土壤含水量空间分布，提高了估计的稳健性并能够评估不确定性。

### 背景

电阻率层析成像技术越来越多地用于研究地下水文过程，在估计土壤含水量方面显示出巨大潜力，但将电阻率信号转换为含水量很复杂，需要开发更稳健的方法并评估不确定性。

### 目的

开发一种方法，提高电阻率层析成像估计土壤含水量的稳健性，同时评估估计中的不确定性。

### 方法

提出集合方法电阻率层析成像技术，结合时间推移ERT数据和基于点的原位土壤含水量测量，通过构建多个反演模型的集合来避免参数选择问题，并利用模型间的变异系数评估不确定性。

### 主要发现

该方法在法国南部两个不同场地的测试中表现良好，计算值与原位测量值吻合，同时能够识别高不确定性区域，提供更全面的信息。

### 结论

集合方法电阻率层析成像提供了一种稳健且自动化的方法将ERT数据转换为相关参数，有助于改进对地下过程的监测和理解。

### 翻译

电阻率层析成像技术越来越多地用于研究地下水文过程。它在估计土壤含水量方面显示出巨大潜力，土壤含水量是一个难以量化的关键属性。然而，将电阻率信号转换为含水量很复杂。这促使人们开发方法来提高估计的稳健性，同时便于评估不确定性。在本文中，我们提出了一种创新方法，称为集合方法电阻率层析成像，构建基于野外数据校准的电阻率集合模型，然后将其转换为含水量的空间分布。该方法结合了时间推移ERT数据和基于点的原位土壤含水量测量。它能够i)通过评估大量模型的性能来避免反演参数选择，ii)通过计算集合中模型间的变异系数来估计最终模型的不确定性，以及iii)将电阻率模型转换为含水量。该方法在法国南部两个不同的野外场地进行了测试。对于每个场地，从多次反演构建的集合模型被选中并转换为土壤含水量。计算值与原位测量值吻合良好，差异较小。识别出了高不确定性区域，提供了比反演代码更经典指标更互补的信息。集合方法电阻率层析成像提供了一种稳健且自动化的方法将ERT数据转换为相关参数，有助于改进对地下过程的监测和理解。


### 论文摘要

Electrical Resistivity Tomography (ERT) is increasingly used to study subsurface hydrological processes. It shows promising potential for estimating soil water content, a key but challenging property to quantify. However, converting the resistivity signal into water content is complex. This encourages developing approaches to increase the robustness of estimates while facilitating the evaluation of uncertainties. In this paper, we propose an innovative method, called the Ensemble Approach ERT (EA-ERT), to build an ensemble model of electrical resistivity calibrated from field data and then to convert it into a spatial distribution of water content. This approach combines time-lapse ERT data with point-based in-situ soil water content measurements. It enables i) circumventing inversion parameter choice by evaluating the performance of a large number of models, ii) estimating uncertainty in the final model by calculating the coefficient of variation among the models composing the ensemble, and iii) converting electrical resistivity models to water content. The method was tested at two dissimilar field sites in southern France. For each site, an ensemble model, built from multiple inversions, was selected and converted into soil water content. The calculated values showed a good fit, with small differences compared to in-situ measurements. Areas of high uncertainty were identified, providing complementary information to the more classical indicators from the inversion code. EA-ERT provides a robust and automatable method to convert ERT data to related parameters, contributing to improved monitoring and understanding processes in the subsurface.

---

## 40. Know-Show: Benchmarking Video-Language Models on Spatio-Temporal Grounded Reasoning

**论文链接:** [http://arxiv.org/abs/2512.05513v1](http://arxiv.org/abs/2512.05513v1)

**作者:** Chinthani Sugandhika, Chen Li, Deepu Rajan, Basura Fernando

**发布时间:** 2025-12-05

### GPT解析

### 总结

Know-Show是一个新的基准测试，用于评估视频语言模型在时空基础推理方面的能力，揭示了当前模型与人类推理之间的显著差距。作者还提出了GRAM插件来增强模型的细粒度基础能力。

### 背景

大型视频语言模型（Video-LMs）在多模态理解方面已取得显著进展，但它们的空间和时间推理能力仍然较弱，缺乏基于视觉和时间证据的坚实基础。

### 目的

开发Know-Show基准测试，评估模型在时空基础推理方面的能力，即模型能够基于视觉和时间证据进行推理的能力，同时统一推理和定位评估框架。

### 方法

Know-Show整合了五个互补场景，涵盖空间（人物、物体、人物-物体、手-物体）和时间维度，基于Charades、Action Genome和Ego4D构建，包含2.5K个人类编写的问题。为弥合差距，作者提出了GRAM插件，通过基于注意力的视频令牌选择和显式时间戳编码增强Video-LMs的细粒度基础能力。

### 主要发现

现有模型难以'展示其所知'，反之亦然，特别是在细粒度的手-物体交互方面。Know-Show基准测试揭示了当前Video-LMs与人类推理之间的显著差距。

### 结论

Know-Show为视频语言理解中的基础推理建立了统一标准，并为开发可解释和可靠的多模态推理系统提供了见解，代码将在https://github.com/LUNAProject22/Know-Show发布。

### 翻译

大型视频语言模型（Video-LMs）在多模态理解方面取得了令人印象深刻的进展，但它们的空间和时间推理仍然缺乏坚实基础。我们提出了Know-Show，这是一个新的基准测试，用于评估时空基础推理能力，即模型能够基于视觉和时间证据进行推理的能力。Know-Show将推理和定位统一在一个评估框架中，包含空间（人物、物体、人物-物体和手-物体）和时间维度上的五个互补场景。该基准基于Charades、Action Genome和Ego4D构建，包含2.5K个人类编写的问题，揭示了当前Video-LMs与人类推理之间的显著差距。为弥合这一差距，我们提出了GRAM，一个无需训练的插件，通过基于注意力的视频令牌选择和显式时间戳编码来增强Video-LMs的细粒度基础能力。在开放和封闭Video-LMs（Qwen、VideoLLaVA、GPT-4o和Gemini等）上的广泛实验表明，现有模型难以'展示其所知'，反之亦然，特别是在细粒度的手-物体交互方面。Know-Show为评估视频语言理解中的基础推理建立了统一标准，并为开发可解释和可靠的多模态推理系统提供了见解。我们将在https://github.com/LUNAProject22/Know-Show发布代码。


### 论文摘要

Large Video-Language Models (Video-LMs) have achieved impressive progress in multimodal understanding, yet their reasoning remains weakly grounded in space and time. We present Know-Show, a new benchmark designed to evaluate spatio-temporal grounded reasoning, the ability of a model to reason about actions and their semantics while simultaneously grounding its inferences in visual and temporal evidence. Know-Show unifies reasoning and localization within a single evaluation framework consisting of five complementary scenarios across spatial (person, object, person-object, and hand-object) and temporal dimensions. Built from Charades, Action Genome, and Ego4D with 2.5K human-authored questions, the benchmark exposes significant gaps between current Video-LMs and human reasoning. To bridge this gap, we propose GRAM, a training-free plug-in that augments Video-LMs with fine-grained grounding through attention-based video token selection and explicit timestamp encoding. Extensive experiments across open and closed Video-LMs (Qwen, VideoLLaVA, GPT-4o, and Gemini, etc.) reveal that existing models struggle to "show what they know" and vice versa, especially in fine-grained hand-object interactions. Know-Show establishes a unified standard for assessing grounded reasoning in video-language understanding and provides insights toward developing interpretable and reliable multimodal reasoning systems. We will release the code at https://github.com/LUNAProject22/Know-Show.

---

## 41. Tree Thinking in the Genomic Era: Unifying Models Across Cells, Populations, and Species

**论文链接:** [http://arxiv.org/abs/2512.05499v1](http://arxiv.org/abs/2512.05499v1)

**作者:** Yun Deng, Shing H. Zhan, Yulin Zhang, Chao Zhang, Bingjie Chen

**发布时间:** 2025-12-05

### GPT解析

### 总结

这篇论文探讨了基于树模型在生物学各尺度(从细胞到种群和物种)中的应用，以及这些方法面临的共同挑战和交叉融合的机会。

### 背景

基因组序列数据的爆炸性增长正在改变我们重建和理解生物系统历史的方式。基于树模型提供了一个表示谱系关系的共同框架，从传统的系统发育学到现在的群体基因组学和细胞生物学。

### 目的

通过比较基于树的方法在不同生物学尺度上的应用，确定统一它们的概念相似性和各领域的独特挑战，为算法创新提供新视角。

### 方法

考察基于树的方法在细胞、种群和物种层面的应用，分析其概念相似性和独特挑战。

### 主要发现

基于树的方法在不同生物学尺度上共享核心统计和算法挑战，包括从基因组信息推断分支历史、整合时空信号、连接谱系结构与进化功能过程。

### 结论

认识到这些方法的共同基础为传统孤立研究领域的交叉融合提供了机会，比较分析可为算法创新和更强大的推断策略提供新视角。

### 翻译

基因组序列数据的持续爆炸正在改变我们重建和理解生物系统历史的方式。跨越生物学尺度，从单个细胞到种群和物种，基于树模型提供了一个表示谱系的共同框架。 once仅限于物种系统发育学，'树思维'现在已深入到群体基因组学和细胞生物学中，揭示了生物体内和生物间的遗传和表型变异的谱系结构。最近，基于树的方法在方法论和计算方面取得了重大进展，包括推断种群祖先重组图的方法、比较基因组学的系统发育框架以及发育和癌症生物学中的谱系追踪技术。尽管数据类型和生物背景不同，但这些方法共享核心的统计和算法挑战：从基因组信息中有效推断分支历史、整合时空信号、将谱系结构与进化和功能过程联系起来。认识到这些共同基础为传统上孤立研究的领域之间的交叉融合提供了机会。通过研究基于树的方法如何在细胞、种群和物种层面应用，我们确定了统一它们的概念相似性以及每个领域面临的独特挑战。这些比较提供了新的视角，可以为算法创新提供信息，并导致更强大的全谱系生物系统的推断策略。


### 论文摘要

The ongoing explosion of genome sequence data is transforming how we reconstruct and understand the histories of biological systems. Across biological scales, from individual cells to populations and species, trees-based models provide a common framework for representing ancestry. Once limited to species phylogenetics, "tree thinking" now extends deeply to population genomics and cell biology, revealing the genealogical structure of genetic and phenotypic variation within and across organisms. Recently, there have been great methodological and computational advances on tree-based methods, including methods for inferring ancestral recombination graphs in populations, phylogenetic frameworks for comparative genomics, and lineage-tracing techniques in developmental and cancer biology. Despite differences in data types and biological contexts, these approaches share core statistical and algorithmic challenges: efficiently inferring branching histories from genomic information, integrating temporal and spatial signals, and connecting genealogical structures to evolutionary and functional processes. Recognizing these shared foundations opens opportunities for cross-fertilization between fields that are traditionally studied in isolation. By examining how tree-based methods are applied across cellular, population, and species scales, we identify the conceptual parallels that unite them and the distinct challenges that each domain presents. These comparisons offer new perspectives that can inform algorithmic innovations and lead to more powerful inference strategies across the full spectrum of biological systems.

---

## 42. Concept-based Explainable Data Mining with VLM for 3D Detection

**论文链接:** [http://arxiv.org/abs/2512.05482v1](http://arxiv.org/abs/2512.05482v1)

**作者:** Mai Tsujimoto

**发布时间:** 2025-12-05

**备注:** 28 pages including appendix. Code: https://github.com/mm1129/concept_based_rare_detector_2025

### GPT解析

### 总结

本文提出了一种新颖的跨模态框架，利用2D视觉语言模型(VLMs)识别和挖掘驾驶场景中的稀有物体，以提高3D物体检测性能。通过整合多种技术并采用概念引导的数据挖掘策略，该框架显著减少了标注负担，同时仅使用少量训练数据就能提升模型性能，特别是在处理拖车和自行车等具有挑战性的物体类别时。

### 背景

稀有物体检测在自动驾驶系统中仍然是一个挑战，尤其是在仅依赖点云数据的情况下。虽然视觉语言模型在图像理解方面表现出强大能力，但其通过智能数据挖掘增强3D物体检测的潜力尚未被充分探索。

### 目的

开发一个利用2D VLMs识别和挖掘驾驶场景中稀有物体的跨模态框架，以提高3D物体检测性能，同时减少标注负担。

### 方法

整合物体检测、语义特征提取、降维和多方面异常检测技术，构建一个连贯且可解释的流程。结合Isolation Forest和t-SNE-based异常检测方法与概念过滤，提取和标注特定稀有物体概念（如施工车辆、摩托车和路障）。

### 主要发现

在nuScenes数据集上的实验表明，概念引导的数据挖掘策略提高了3D物体检测模型性能，仅使用一小部分训练数据就实现了显著提升，特别是在拖车和自行车等挑战性物体类别上，相比使用相同数量的随机数据，改进更为明显。

### 结论

该研究对安全关键型自动驾驶系统中数据集的高效构建具有重要意义，提供了一种减少标注负担同时提高模型性能的有效方法。

### 翻译

稀有物体检测在自动驾驶系统中仍然是一个具有挑战性的任务，尤其是在仅依赖点云数据的情况下。虽然视觉语言模型(VLMs)在图像理解方面表现出强大能力，但其通过智能数据挖掘增强3D物体检测的潜力尚未被充分探索。本文提出了一种新颖的跨模态框架，利用2D VLMs来识别和挖掘驾驶场景中的稀有物体，从而提高3D物体检测性能。我们的方法整合了物体检测、语义特征提取、降维和多方面异常检测等互补技术，形成一个连贯且可解释的流程，系统性地识别驾驶场景中稀有但关键的物体。通过将Isolation Forest和基于t-SNE的异常检测方法与基于概念的过滤相结合，该框架有效地识别出具有语义意义的稀有物体。这种方法的一个关键优势在于能够提取和标注特定的稀有物体概念，如施工车辆、摩托车和路障。这显著减少了标注负担，只专注于最有价值的训练样本。在nuScenes数据集上的实验表明，这种概念引导的数据挖掘策略提高了3D物体检测模型的性能，同时仅使用一小部分训练数据，特别是在拖车和自行车等具有挑战性的物体类别上，相比使用相同数量的随机数据，改进尤为显著。这一发现对安全关键型自动驾驶系统中数据集的高效构建具有重要意义。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自动驾驶系统中稀有目标检测的挑战，特别是当仅依赖点云数据时。这个问题在现实中非常重要，因为稀有对象（如摩托车、自行车、施工车辆）对自动驾驶安全至关重要，但这些对象在训练数据中出现的频率有限，导致检测性能不佳。传统方法通过收集和标注额外数据来提高性能，但这种方法耗时且昂贵。此外，自动驾驶数据集（如nuScenes）存在严重的类别不平衡问题，常见对象（如汽车和行人）远多于稀有对象，这给开发鲁棒检测系统带来了巨大挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到视觉语言模型（VLMs）在图像理解方面具有强大能力，但其在通过智能数据挖掘增强3D目标检测方面的潜力尚未被充分探索。他们注意到仅使用点云数据进行稀有对象检测的局限性，因此提出利用2D VLMs来识别和挖掘驾驶场景中的稀有对象。作者借鉴了多种现有工作：使用CLIP进行特征提取，Qwen2-VL进行图像字幕生成，YOLOv8进行对象检测，以及结合Isolation Forest和t-SNE进行异常检测。此外，还参考了概念瓶颈模型（CBMs）用于增强模型解释性，以及基础模型进行概念提取和高效检测架构。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用视觉语言模型（VLM）的语义理解能力来识别和挖掘稀有对象，通过概念过滤和异常检测相结合的方法，专注于有意义的稀有对象，减少标注负担，只关注最有价值的训练样本，创建一个可解释的框架。整体实现流程包括三个主要组件：1) 对象概念嵌入系统：使用YOLOv8检测对象，CLIP提取特征嵌入，结合t-SNE和Isolation Forest进行异常检测；2) 稀有对象挖掘方法：组合异常检测结果，基于概念过滤；3) 目标数据挖掘框架：选择包含特定稀有对象概念的场景进行标注，提高欠表示类别的检测性能。具体步骤包括检测对象、提取特征、异常检测、概念生成、分类筛选和构建训练数据集。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 跨模态框架：首次将2D视觉语言理解与3D检测通过基础模型桥接；2) 概念驱动的方法：利用VLM的语义理解能力而非仅几何特征识别稀有对象；3) 可解释性：提供透明的、基于概念的数据选择决策解释；4) 高效性能提升：仅使用20%的训练数据实现显著的稀有对象类别性能提升；5) 实用安全贡献：针对安全关键对象的针对性改进。相比之前工作的不同在于：与传统数据挖掘方法不同，该方法利用VLM的语义理解能力；与仅使用几何特征的方法相比，能更好地理解复杂对象的语义内容；与硬例挖掘方法不同，基于语义概念而非简单难度指标；与概念瓶颈模型不同，专注于数据挖掘而非模型解释；与VLMine相比，更加注重可解释性和概念驱动的数据选择。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于概念的可解释数据挖掘框架，利用视觉语言模型智能识别和选择训练样本，显著提高了3D目标检测中稀有类别的检测性能，同时大幅降低了标注成本。'}


### 论文摘要

Rare-object detection remains a challenging task in autonomous driving systems, particularly when relying solely on point cloud data. Although Vision-Language Models (VLMs) exhibit strong capabilities in image understanding, their potential to enhance 3D object detection through intelligent data mining has not been fully explored. This paper proposes a novel cross-modal framework that leverages 2D VLMs to identify and mine rare objects from driving scenes, thereby improving 3D object detection performance. Our approach synthesizes complementary techniques such as object detection, semantic feature extraction, dimensionality reduction, and multi-faceted outlier detection into a cohesive, explainable pipeline that systematically identifies rare but critical objects in driving scenes. By combining Isolation Forest and t-SNE-based outlier detection methods with concept-based filtering, the framework effectively identifies semantically meaningful rare objects. A key strength of this approach lies in its ability to extract and annotate targeted rare object concepts such as construction vehicles, motorcycles, and barriers. This substantially reduces the annotation burden and focuses only on the most valuable training samples. Experiments on the nuScenes dataset demonstrate that this concept-guided data mining strategy enhances the performance of 3D object detection models while utilizing only a fraction of the training data, with particularly notable improvements for challenging object categories such as trailers and bicycles compared with the same amount of random data. This finding has substantial implications for the efficient curation of datasets in safety-critical autonomous systems.

---

## 43. Unleashing Temporal Capacity of Spiking Neural Networks through Spatiotemporal Separation

**论文链接:** [http://arxiv.org/abs/2512.05472v1](http://arxiv.org/abs/2512.05472v1)

**作者:** Yiting Dong, Zhaofei Yu, Jianhao Ding, Zijie Xu, Tiejun Huang

**发布时间:** 2025-12-05

### GPT解析

### 总结

该研究探讨了脉冲神经网络(SNNs)中膜电位传播在时序处理中的实际贡献，发现适度移除膜传播可提高性能，并提出了空间-时间可分离网络(STSep)模型，在视频理解任务中取得了优越性能。

### 背景

脉冲神经网络(SNNs)被认为自然适合时序处理，膜电位传播被广泛认为是核心时序建模机制，但现有研究缺乏对其在复杂时序任务中实际贡献的分析。

### 目的

设计非状态(NS)模型，逐步移除膜传播机制，以量化膜电位传播在SNNs各阶段的实际作用。

### 方法

通过设计非状态模型逐步移除膜传播，基于发现的空间-时间资源竞争现象，提出空间-时间可分离网络(STSep)，将残差块解耦为独立的空间分支(专注于语义提取)和时间分支(通过显式时间差异捕捉运动)。

### 主要发现

存在反直觉现象：在浅层或深层适度移除膜传播可提高性能，而过度移除则导致性能崩溃；这归因于空间-时间资源竞争，神经元在有限范围内同时编码语义和动态，时间状态消耗了空间学习的能力。

### 结论

提出的STSep在Something-Something V2、UCF101和HMDB51数据集上实现了优越性能，检索任务和注意力分析证实了模型专注于运动而非静态外观，为SNNs的时序机制提供了新视角，并为视频理解中的时空建模提供了有效解决方案。

### 翻译

脉冲神经网络(SNNs)被认为自然适合时序处理，膜电位传播被广泛认为是核心时序建模机制。然而，现有研究缺乏对其在复杂时序任务中实际贡献的分析。我们设计了非状态(NS)模型，逐步移除膜传播以量化其在各阶段的作用。实验揭示了一种反直觉现象：在浅层或深层适度移除可提高性能，而过度移除则会导致崩溃。我们将此归因于空间-时间资源竞争，神经元在有限范围内同时编码语义和动态，时间状态消耗了空间学习的能力。基于此，我们提出空间-时间可分离网络(STSep)，将残差块解耦为独立的空间和时间分支。空间分支专注于语义提取，而时间分支通过显式时间差异捕捉运动。在Something-Something V2、UCF101和HMDB51上的实验显示STSep实现了优越性能，检索任务和注意力分析证实了模型专注于运动而非静态外观。这项工作为SNNs的时序机制提供了新视角，并为视频理解中的时空建模提供了有效解决方案。


### 论文摘要

Spiking Neural Networks (SNNs) are considered naturally suited for temporal processing, with membrane potential propagation widely regarded as the core temporal modeling mechanism. However, existing research lack analysis of its actual contributions in complex temporal tasks. We design Non-Stateful (NS) models progressively removing membrane propagation to quantify its stage-wise role. Experiments reveal a counterintuitive phenomenon: moderate removal in shallow or deep layers improves performance, while excessive removal causes collapse. We attribute this to spatio-temporal resource competition where neurons encode both semantics and dynamics within limited range, with temporal state consuming capacity for spatial learning. Based on this, we propose Spatial-Temporal Separable Network (STSep), decoupling residual blocks into independent spatial and temporal branches. The spatial branch focuses on semantic extraction while the temporal branch captures motion through explicit temporal differences. Experiments on Something-Something V2, UCF101, and HMDB51 show STSep achieves superior performance, with retrieval task and attention analysis confirming focus on motion rather than static appearance. This work provides new perspectives on SNNs' temporal mechanisms and an effective solution for spatiotemporal modeling in video understanding.

---

## 44. From Vision to Touch: Bridging Visual and Tactile Principles for Accessible Data Representation

**论文链接:** [http://arxiv.org/abs/2512.05433v1](http://arxiv.org/abs/2512.05433v1)

**作者:** Kim Marriott, Matthew Butler, Leona Holloway, Bill Jolley, Bongshin Lee, Bruce Maguire, Danielle Albers Szafir

**发布时间:** 2025-12-05

**备注:** To be published by IEEE as part of the 2025 Visualization Conference (VIS)

### GPT解析

### 总结

该研究探讨了触觉图形对盲人和低视力人群的益处，建立了触觉信息图形设计的理论框架。

### 背景

触觉图形被广泛用于向盲人和低视力人群展示地图和统计图表，无障碍指南建议在空间关系重要的图形中使用触觉图形，随着商用可刷新触觉显示器的出现，其使用预计将增长。

### 目的

解决对精心设计的触觉信息图形相比文本描述对盲人和低视力人群益处理解不足的问题。

### 方法

引入一个考虑编码、感知和认知三个组成部分的框架，检查视觉信息图形的已知益处，并探索这些益处对触觉信息图形的适用性。

### 主要发现

建立了触觉信息图形设计的初步理论基础，确定了未来研究方向。

### 结论

为信息图形的触觉优先设计建立了初步理论基础，并识别了未来研究途径。

### 翻译

触觉图形被广泛用于向盲人和低视力人群展示地图和统计图表，无障碍指南建议在空间关系重要的图形中使用触觉图形。随着商用可刷新触觉显示器的出现，其使用预计将增长。然而，与视觉信息图形形成鲜明对比的是，我们缺乏对精心设计的触觉信息图形相比文本描述对盲人和低视力人群益处的清晰理解。为解决这一空白，我们引入了一个考虑编码、感知和认知三个组成部分的框架，来检查视觉信息图形的已知益处并探索它们对触觉信息图形的适用性。这项工作为信息图形的触觉优先设计建立了初步理论基础，并确定了未来研究途径。


### 论文摘要

Tactile graphics are widely used to present maps and statistical diagrams to blind and low vision (BLV) people, with accessibility guidelines recommending their use for graphics where spatial relationships are important. Their use is expected to grow with the advent of commodity refreshable tactile displays. However, in stark contrast to visual information graphics, we lack a clear understanding of the benefits that well-designed tactile information graphics offer over text descriptions for BLV people. To address this gap, we introduce a framework considering the three components of encoding, perception and cognition to examine the known benefits for visual information graphics and explore their applicability to tactile information graphics. This work establishes a preliminary theoretical foundation for the tactile-first design of information graphics and identifies future research avenues.

---

## 45. The Dynamic Prior: Understanding 3D Structures for Casual Dynamic Videos

**论文链接:** [http://arxiv.org/abs/2512.05398v1](http://arxiv.org/abs/2512.05398v1)

**作者:** Zhuoyuan Wu, Xurui Yang, Jiahui Huang, Yue Wang, Jun Gao

**发布时间:** 2025-12-05

**备注:** Code is available at https://github.com/wuzy2115/DYNAPO

### GPT解析

### 总结

本文提出了一种名为Dynamic Prior的新方法，无需特定任务训练即可稳健识别视频中的动态物体，从而提高3D场景重建的准确性。

### 背景

从野外视频中准确估计相机姿态、3D场景几何和物体运动是传统运动恢复结构管道的长期挑战，主要因为存在动态物体。

### 目的

开发一种无需大规模运动分割数据集训练的方法，能够准确识别动态物体，提高3D结构理解的准确性和鲁棒性。

### 方法

引入Dynamic Prior方法，结合视觉语言模型(VLMs)的强大推理能力和SAM2的细粒度空间分割能力，无需特定任务训练即可稳健识别动态物体，并可无缝集成到最先进的管道中用于相机姿态优化、深度重建和4D轨迹估计。

### 主要发现

在合成和真实世界视频上的实验表明，Dynamic Prior不仅在运动分割方面达到了最先进的性能，还显著提高了3D结构理解的准确性和鲁棒性。

### 结论

Dynamic Prior方法有效解决了传统运动恢复结构管道在处理动态物体时的局限性，为从野外视频中准确估计相机姿态、3D场景几何和物体运动提供了新的解决方案。

### 翻译

从野外视频中准确估计相机姿态、3D场景几何和物体运动是传统运动恢复结构管道的长期挑战，因为存在动态物体。最近基于学习的方法尝试通过训练运动估计器来过滤动态物体并专注于静态背景，但它们的性能很大程度上受限于大规模运动分割数据集的可用性，导致分割不准确，进而影响3D结构理解的准确性。在这项工作中，我们引入了Dynamic Prior，无需特定任务训练即可稳健识别动态物体，利用了视觉语言模型的强大推理能力和SAM2的细粒度空间分割能力。Dynamic Prior可以无缝集成到最先进的管道中，用于相机姿态优化、深度重建和4D轨迹估计。在合成和真实世界视频上的大量实验表明，Dynamic Prior不仅在运动分割方面达到了最先进的性能，还显著提高了3D结构理解的准确性和鲁棒性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从包含动态物体的日常视频中准确估计相机姿态、3D场景几何和物体运动的问题。这个问题在现实中非常重要，因为动态物体在真实世界视频中无处不在，而传统方法（如结构从运动）在静态场景中表现良好但在动态场景中会失败。准确的3D结构理解对VR/AR、机器人、自动驾驶和空间智能等应用至关重要，现有方法由于依赖大规模运动分割数据集进行训练，导致泛化能力有限，无法很好地处理真实世界中的动态场景。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到现有方法受限于大规模运动分割数据集的可用性，同时注意到大型视觉基础模型（如VLMs和SAM2）在视觉感知方面的成功。他们设计了一种结合VLMs的高层次推理能力和SAM2的细粒度空间分割能力的框架。具体来说，他们借鉴了VLMs（如GPT-4o）的视频理解能力和SAM2的视频分割能力，并参考了现有的3D结构理解框架（如AnyCam、MegaSam、Stereo4D）进行集成应用。这种方法不需要特定任务训练，而是利用预训练模型的通用能力来提高泛化性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用视觉语言模型（VLMs）的推理能力和SAM2的分割能力，无需特定任务训练就能准确识别动态物体，生成精确的动态物体掩码，用于优化相机姿态、深度重建和4D轨迹估计。整体流程分为两个阶段：1）动态物体推理阶段：从视频中采样关键帧，使用VLMs通过思维链过程分析关键帧，识别动态物体，并为每个动态物体选择最显著的关键帧并生成描述；2）动态物体分割阶段：使用Sa2VA在选定关键帧上分割每个动态物体，利用SAM2的记忆机制将掩码传播到整个视频序列，合并所有动态物体的掩码生成最终的二进制掩码序列。最后将此掩码应用到3D结构理解任务中。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）提出Dynamic Prior框架，利用VLMs和SAM2的先验知识进行动态物体识别，无需特定任务训练；2）结合VLMs的高层次推理能力和SAM2的细粒度分割能力；3）将动态掩码无缝集成到多种3D结构理解框架中；4）通过思维链过程进行动态物体推理，提供更深入、上下文感知的动态场景理解。相比之前的工作，这种方法不依赖大规模运动分割数据集进行训练，能更好地理解复杂动态场景，提供更准确、更清晰的动态物体掩码，而不是模糊的不确定性图，在多个任务上实现了性能提升。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了Dynamic Prior框架，通过结合视觉语言模型的推理能力和SAM2的分割能力，实现了无需训练的精确动态物体识别，显著提升了从日常动态视频中理解3D结构的准确性和鲁棒性。'}


### 论文摘要

Estimating accurate camera poses, 3D scene geometry, and object motion from in-the-wild videos is a long-standing challenge for classical structure from motion pipelines due to the presence of dynamic objects. Recent learning-based methods attempt to overcome this challenge by training motion estimators to filter dynamic objects and focus on the static background. However, their performance is largely limited by the availability of large-scale motion segmentation datasets, resulting in inaccurate segmentation and, therefore, inferior structural 3D understanding. In this work, we introduce the Dynamic Prior (\ourmodel) to robustly identify dynamic objects without task-specific training, leveraging the powerful reasoning capabilities of Vision-Language Models (VLMs) and the fine-grained spatial segmentation capacity of SAM2. \ourmodel can be seamlessly integrated into state-of-the-art pipelines for camera pose optimization, depth reconstruction, and 4D trajectory estimation. Extensive experiments on both synthetic and real-world videos demonstrate that \ourmodel not only achieves state-of-the-art performance on motion segmentation, but also significantly improves accuracy and robustness for structural 3D understanding.

---

## 46. Elevation- and Tilt-Aware Shadow Fading Correlation Modeling for UAV Communications

**论文链接:** [http://arxiv.org/abs/2512.05332v1](http://arxiv.org/abs/2512.05332v1)

**作者:** Mushfiqur Rahman, Ismail Guvenc, Mihail Sichitiu, Jason A. Abrahamson, Bryton J. Petersen, Amitabh Mishra, Arupjyoti Bhuyan

**发布时间:** 2025-12-05

**备注:** Asilomar Conference on Signals, Systems, and Computers (2025), 7 pages

### GPT解析

### 总结

传统阴影衰落相关模型依赖空间距离，忽略了无人机的3D方向和仰角。即使俯仰角有微小变化（5到10度）也会显著影响无人机观察到的信号强度。研究提出了一个仰角和倾斜角感知的空间相关模型，并通过实验验证了其有效性。

### 背景

未来无线网络需要更准确地理解信道行为以实现高效通信并减少干扰。无人机在这些网络中将发挥重要作用，提供多样化的应用和灵活的部署选项。准确描述无人机通信中的阴影衰落行为仍然是一个挑战。

### 目的

研究无人机俯仰角和仰角几何形状对阴影衰落的影响，并提出一个考虑仰角和倾斜角的空间相关模型。

### 方法

使用在3.32 GHz频率、125 kHz带宽下，在农村环境中收集的真实世界固定高度无人机测量数据集。将提出的相关模型整合到普通克里金框架中进行信号强度预测。

### 主要发现

10度的倾斜角分离和20度的仰角分离可以分别使阴影衰落相关性降低高达15%和40%。与忽略无人机方向和仰角的传统相关模型相比，将提出的模型整合到OK框架中可以使中值RMSE改善约1.5 dB。

### 结论

无人机的3D方向和仰角对信号强度有显著影响。提出的考虑仰角和倾斜角的空间相关模型比传统模型更准确。

### 翻译

未来无线网络需要更准确地理解信道行为，以实现高效通信并减少干扰。无人机（UAV）在这些网络中将发挥重要作用，提供多样化的应用和灵活的部署选项。然而，准确描述无人机通信中的阴影衰落（SF）行为仍然是一个挑战。传统的SF相关模型依赖空间距离，忽略了无人机的3D方向和仰角。即使俯仰角有微小变化（5到10度）也会显著影响无人机观察到的信号强度。在本研究中，我们研究了无人机俯仰和仰角几何形状对SF的影响，并提出一个考虑仰角和倾斜角的空间相关模型。我们使用了一个在3.32 GHz频率、125 kHz带宽下，在农村环境中收集的真实世界固定高度无人机测量数据集。结果表明，10度的倾斜角分离和20度的仰角分离可以分别使SF相关性降低高达15%和40%。此外，将提出的相关模型整合到普通克里金（OK）框架中进行信号强度预测，与忽略无人机方向和仰角的传统相关模型相比，可以使中值RMSE改善约1.5 dB。


### 论文摘要

Future wireless networks demand a more accurate understanding of channel behavior to enable efficient communication with reduced interference. Uncrewed Aerial Vehicles (UAVs) are poised to play an integral role in these networks, offering versatile applications and flexible deployment options. However, accurately characterizing the shadow fading (SF) behavior in UAV communications remains a challenge. Traditional SF correlation models rely on spatial distance and neglect the UAV's 3D orientation and elevation angle. Yet even slight variations in pitch angle (5 to 10 degrees) can significantly affect the signal strength observed by a UAV. In this study, we investigate the impact of UAV pitch and elevation geometry on SF and propose an elevation- and tilt-aware spatial correlation model. We use a real-world fixed-altitude UAV measurement dataset collected in a rural environment at 3.32 GHz with a 125 kHz bandwidth. Results show that a 10-degree tilt-angle separation and a 20-degree elevation-angle separation can reduce the SF correlation by up to 15% and 40%, respectively. In addition, integrating the proposed correlation model into the ordinary Kriging (OK) framework for signal strength prediction yields an approximate 1.5 dB improvement in median RMSE relative to the traditional correlation model that ignores UAV orientation and elevation.

---

## 47. From Segments to Scenes: Temporal Understanding in Autonomous Driving via Vision-Language Model

**论文链接:** [http://arxiv.org/abs/2512.05277v1](http://arxiv.org/abs/2512.05277v1)

**作者:** Kevin Cannons, Saeed Ranjbar Alvar, Mohammad Asiful Hossain, Ahmad Rezaei, Mohsen Gholami, Alireza Heidarikhazaei, Zhou Weimin, Yong Zhang, Mohammad Akbari

**发布时间:** 2025-12-04

### GPT解析

### 总结

这项研究提出了专注于自动驾驶时间理解的新基准测试TAD，评估了当前最先进模型的表现，并提出了两种无需训练的改进方法Scene-CoT和TCogMap，显著提高了时间理解能力。

### 背景

自动驾驶中的时间理解仍然是一个重大挑战，即使是最新的视觉语言模型也面临这一问题。现有数据集和基准测试侧重于其他视频内容，如体育、烹饪和电影，没有专门针对自动驾驶第一人称视频时间理解独特挑战的基准测试。

### 目的

填补自动驾驶时间理解研究空白，提出TAD基准测试，评估视觉语言模型捕捉自动驾驶中动作之间动态关系的能力。

### 方法

创建了包含近6,000个问答对、涵盖7个人类设计任务的TAD基准测试；评估了9个通用模型和最先进AD专用模型；提出了两种无需训练的解决方案：Scene-CoT（利用思维链）和TCogMap（结合以自我为中心的时间认知图），并将其与现有VLMs集成。

### 主要发现

当前最先进模型在TAD上表现不佳，主要原因是细粒度运动理解不完善；提出的改进方法将TAD上的平均准确率提高了高达17.72%。

### 结论

通过引入TAD、基准测试多个最先进模型和提出有效改进，旨在促进未来对自动驾驶中时间理解的研究。基准测试和评估代码分别在Hugging Face和Github上提供。

### 翻译

自动驾驶中的时间理解仍然是一个重大挑战，即使是最新的视觉语言模型也面临这一问题。先前的工作引入了旨在提高时间推理能力的数据集和基准测试，但这些数据集和基准测试侧重于其他视频内容，包括体育、烹饪和电影。没有现有基准测试专门专注于自动驾驶第一人称视频中的时间理解独特挑战。为填补这一空白，本文提出了自动驾驶时间理解(TAD)基准测试，该基准测试评估视觉语言模型捕捉自动驾驶中动作之间动态关系的能力。TAD包含近6,000个问答对，涵盖7个人类设计的任务。此外，还对9个闭源和开源通用模型以及最先进的AD专用模型进行了评估。当应用于TAD时，当前最先进的模型表现不佳，主要原因是细粒度运动理解不完善。为提高运动理解能力和TAD上的整体准确率，提出了两种新的无需训练的解决方案：Scene-CoT，它利用思维链(Chain-of-Thought)，以及TCogMap，它结合了以自我为中心的时间认知图。将所提出的方法与现有视觉语言模型集成后，在TAD上的平均准确率提高了高达17.72%。通过引入TAD、基准测试多个最先进模型和提出有效改进，这项工作旨在促进未来对自动驾驶中时间理解的研究。该基准测试和评估代码分别在Hugging Face和Github上提供。


### 论文摘要

Temporal understanding in autonomous driving (AD) remains a significant challenge, even for recent state-of-the-art (SoTA) Vision-Language Models (VLMs). Prior work has introduced datasets and benchmarks aimed at improving temporal reasoning, but these have emphasized other video content, including sports, cooking, and movies. No existing benchmark focuses exclusively on the unique challenges of temporal understanding in ego-centric AD footage. To fill this gap, the Temporal Understanding in Autonomous Driving (TAD) benchmark is presented, which evaluates VLMs' ability to capture the dynamic relationships between actions in AD. TAD comprises nearly 6,000 question-answer (QA) pairs, spanning 7 human-designed tasks. In addition, an evaluation is performed that consists of 9 closed- and open-source generalist models as well as SoTA AD specialist models. When applied to TAD, current SoTA models demonstrated substandard accuracies, largely due to imperfect fine-grained motion understanding. To improve motion understanding and overall accuracy on TAD, two novel training-free solutions are proposed: Scene-CoT, that leverages Chain-of-Thought (CoT) and TCogMap, which incorporates an ego-centric temporal cognitive map. The proposed approaches are integrated with existing VLMs and improve average accuracy on TAD by up to 17.72%. By introducing TAD, benchmarking multiple SoTA models, and proposing effective enhancements, this work aims to catalyze future research on temporal understanding in AD. The benchmark and evaluation code are available at \href{https://huggingface.co/datasets/vbdai/TAD}{Hugging Face} and \href{https://github.com/vbdi/tad_bench}{Github}, respectively.

---

## 48. A Framework for Quantum Simulations of Energy-Loss and Hadronization in Non-Abelian Gauge Theories: SU(2) Lattice Gauge Theory in 1+1D

**论文链接:** [http://arxiv.org/abs/2512.05210v1](http://arxiv.org/abs/2512.05210v1)

**作者:** Zhiyao Li, Marc Illa, Martin J. Savage

**发布时间:** 2025-12-04

**备注:** 28 pages main text, 16 pages appendices, 34 figures, 14 tables

### GPT解析

### 总结

该研究建立了在量子计算机上模拟非平衡强相互作用物质中能量损失和强子化的框架，并在1+1D SU(2)格子上成功模拟了重夸克穿越轻夸克系统。通过发展非阿贝尔理论与阿贝尔理论模拟的概念进展，实现了对轻夸克能量、局部非阿贝尔电荷密度和多部分纠缠演化的计算，并在IBM量子计算机上验证了结果。

### 背景

模拟能量损失和强子化对于理解非平衡强相互作用物质中的各种现象至关重要。传统计算方法面临挑战，量子计算机为这类复杂系统的模拟提供了新的可能性。

### 目的

建立并验证一个在量子计算机上模拟非平衡强相互作用物质中能量损失和强子化的框架，并应用于重夸克穿越轻夸克系统的具体案例。

### 方法

在1+1D SU(2)格子上模拟重夸克穿越轻夸克系统。将重夸克与轻夸克一起映射到量子比特，限制重夸克在格点间的离散运动。使用强子算子实现色纠缠，应用区域分解进行量子态准备。使用费米子SWAP操作实现重夸克的离散运动。在IBM的量子计算机上使用18个量子比特进行了L=3空间位置的模拟，并应用了一系列错误缓解技术。

### 主要发现

1) 发展了非阿贝尔理论与阿贝尔理论模拟的概念进展；2) 能够计算轻夸克中的能量演化、局部非阿贝尔电荷密度和多部分纠缠；3) 将重夸克映射到量子比特并限制其离散运动是有效的；4) 区域分解在量子态准备中有效；5) 可扩展的量子电路能够处理格子上非阿贝尔电荷部分的异构性；6) 使用费米子SWAP操作实现重夸克的离散运动；7) 在量子计算机上成功进行了模拟，结果与经典模拟一致。

### 结论

所提出的框架成功地在量子计算机上模拟了强相互作用系统中的能量损失和强子化现象。该框架可推广到其他非阿贝尔群，包括量子色动力学的SU(3)，为研究强相互作用物质提供了新的量子计算方法。

### 翻译

能量损失和强子化的模拟对于理解非平衡强相互作用物质中的各种现象至关重要。我们建立了在量子计算机上进行此类模拟的框架，并将其应用于穿越适度大小的1+1D SU(2)轻夸克格子的重夸克。我们发展了关于非阿贝尔理论与阿贝尔理论模拟的概念进展，使得能够计算轻夸克中的能量演化、局部非阿贝尔电荷密度以及它们的多部分纠缠。非阿贝尔电荷算子对任意状态的非平凡作用表明，应将重夸克与轻夸克一起映射到量子比特，并将重夸克的运动限制为空间格点之间的离散步骤。此外，使用强子算子实现了重夸克与轻夸克之间的色纠缠，并且区域分解被证明在量子态准备中是有效的。使用可扩展的量子电路来考虑格子上非阿贝尔电荷部分的异构性，以在重夸克存在的情况下制备相互作用基态波函数。重夸克在相邻空间位置之间的离散运动使用费米子SWAP操作实现。使用IBM的ibm_pittsburgh量子计算机对L=3空间位置的系统动力学进行了量子模拟，使用18个量子比特，其中态准备、运动和一个二阶Trotter时间演化步骤的电路具有398个双量子比特深度。使用一系列错误缓解技术从模拟中提取可观测量，提供了与经典模拟结果一致的结果。这里提出的框架可以 straightforward地推广到其他非阿贝尔群，包括量子色动力学的SU(3)。


### 论文摘要

Simulations of energy loss and hadronization are essential for understanding a range of phenomena in non-equilibrium strongly-interacting matter. We establish a framework for performing such simulations on a quantum computer and apply it to a heavy quark moving across a modest-sized 1+1D SU(2) lattice of light quarks. Conceptual advances with regard to simulations of non-Abelian versus Abelian theories are developed, allowing for the evolution of the energy in light quarks, of their local non-Abelian charge densities, and of their multi-partite entanglement to be computed. The non-trivial action of non-Abelian charge operators on arbitrary states suggests mapping the heavy quarks to qubits alongside the light quarks, and limits the heavy-quark motion to discrete steps among spatial lattice sites. Further, the color entanglement among the heavy quarks and light quarks is implemented using hadronic operators, and Domain Decomposition is shown to be effective in quantum state preparation. Scalable quantum circuits that account for the heterogeneity of non-Abelian charge sectors across the lattice are used to prepare the interacting ground-state wavefunction in the presence of heavy quarks. The discrete motion of heavy quarks between adjacent spatial sites is implemented using fermionic SWAP operations. Quantum simulations of the dynamics of a system on $L=3$ spatial sites are performed using IBM's ${\tt ibm\_pittsburgh}$ quantum computer using 18 qubits, for which the circuits for state preparation, motion, and one second-order Trotter step of time evolution have a two-qubit depth of 398. A suite of error mitigation techniques are used to extract the observables from the simulations, providing results that are in good agreement with classical simulations. The framework presented here generalizes straightforwardly to other non-Abelian groups, including SU(3) for quantum chromodynamics.

---

## 49. COOPER: A Unified Model for Cooperative Perception and Reasoning in Spatial Intelligence

**论文链接:** [http://arxiv.org/abs/2512.04563v2](http://arxiv.org/abs/2512.04563v2)

**作者:** Zefeng Zhang, Xiangzhao Hao, Hengzhu Tang, Zhenyu Zhang, Jiawei Sheng, Xiaodong Li, Zhenyang Li, Li Gao, Daiting Shi, Dawei Yin, Tingwen Liu

**发布时间:** 2025-12-04

### GPT解析

### 总结

该研究提出了COOPERS模型，通过统一的多模态大语言框架同时增强空间感知和推理能力，实现了空间推理性能的提升。

### 背景

视觉空间推理对多模态大语言模型理解物体属性和空间关系至关重要，但当前模型在3D感知推理方面仍有困难。

### 目的

探究统一的MLLM是否能够发展出增强空间感知的能力，并通过自适应交错推理实现更强的空间智能。

### 方法

提出了COOPER模型，利用深度和分割作为辅助模态，通过两个阶段的训练获取辅助模态生成和自适应交错推理能力。

### 主要发现

COOPER在空间推理上实现了平均6.91%的改进，同时保持了通用性能；仅针对辅助模态生成训练的变体在距离和大小估计上获得了7.92%的提升。

### 结论

学习生成辅助模态有助于内化空间知识并加强空间理解，统一的MLLM框架能够同时增强空间感知和推理能力。

### 翻译

视觉空间推理对于使多模态大语言模型能够理解物体属性和空间关系至关重要，但当前模型仍在3D感知推理方面存在困难。现有方法通常通过添加深度和分割等辅助模态来增强感知，或通过空间VQA数据集训练和应用强化学习来增强推理，从而将这两个方面孤立处理。在这项工作中，我们探究了统一的MLLM是否能够发展出增强空间感知的能力，并通过自适应交错推理实现更强的空间智能。我们提出了COOPER，一个统一的MLLM，利用深度和分割作为辅助模态，并通过两个阶段的训练来获取辅助模态生成和自适应交错推理能力。COOPER在空间推理上实现了平均6.91%的改进，同时保持了通用性能。此外，即使仅针对辅助模态生成训练的变体，在距离和大小估计上也获得了7.92%的提升，这表明学习生成辅助模态有助于内化空间知识并加强空间理解。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多模态大语言模型在视觉空间推理方面的局限性，特别是3D感知和推理能力不足的问题。这个问题很重要，因为视觉空间推理是实现人类水平智能的关键一步，对机器人技术、自动驾驶和AR/VR等下游应用至关重要。当前方法通常将感知增强和推理增强视为独立部分，限制了模型在复杂空间任务上的表现。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到感知和推理是空间智能的两个相互依赖支柱，应协同工作而非分离。他们借鉴了统一MLLM架构（特别是BAGEL框架）的潜力，同时识别出两个主要挑战：生成非RGB辅助模态的困难和自适应推理的实现。解决方案包括将辅助模态转换为RGB空间进行训练，以及设计SFT+RL框架来优化推理行为。他们借鉴了多模态链式思维、Rectified Flow生成方法和GRPO强化学习等技术。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个统一模型，能协同结合感知增强和推理增强，通过内在多模态链式思维实现更强空间智能。模型能自己生成深度图和分割图等辅助模态，并自适应决定何时进行感知增强或推理增强。实现流程分两阶段：1）辅助模态生成阶段，将深度和分割转换为RGB伪图像进行训练；2）自适应交错推理阶段，包括数据构建、监督微调获取基础能力、使用CPR奖励进行强化学习优化推理行为。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）提出空间智能的交错推理新范式，协同统一感知和推理；2）设计细粒度训练流程，使模型先学习生成辅助模态，再通过RL获取自适应推理能力；3）在空间推理和通用多模态基准上广泛验证。相比之前工作，COOPER在一个统一模型中协同工作感知和推理，而非将它们视为独立部分；模型能自己生成辅助模态而非依赖外部提供；能自适应决定何时生成何种模态；通过学习生成辅助模态内部化空间知识。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'COOPER通过统一感知和推理在一个多模态大语言模型中，实现了自适应的交错空间智能，显著提升了3D空间理解和推理能力，同时保持了通用多模态性能。'}


### 论文摘要

Visual Spatial Reasoning is crucial for enabling Multimodal Large Language Models (MLLMs) to understand object properties and spatial relationships, yet current models still struggle with 3D-aware reasoning. Existing approaches typically enhance either perception, by augmenting RGB inputs with auxiliary modalities such as depth and segmentation, or reasoning, by training on spatial VQA datasets and applying reinforcement learning, and thus treat these two aspects in isolation. In this work, we investigate whether a unified MLLM can develop an intrinsic ability to enhance spatial perception and, through adaptive interleaved reasoning, achieve stronger spatial intelligence. We propose \textbf{COOPER}, a unified MLLM that leverages depth and segmentation as auxiliary modalities and is trained in two stages to acquire auxiliary modality generation and adaptive, interleaved reasoning capabilities. COOPER achieves an average \textbf{6.91\%} improvement in spatial reasoning while maintaining general performance. Moreover, even a variant trained only for auxiliary modality generation attains a \textbf{7.92\%} gain on distance and size estimation, suggesting that learning to generate auxiliary modalities helps internalize spatial knowledge and strengthen spatial understanding.

---

## 50. EditThinker: Unlocking Iterative Reasoning for Any Image Editor

**论文链接:** [http://arxiv.org/abs/2512.05965v1](http://arxiv.org/abs/2512.05965v1)

**作者:** Hongyu Li, Manyuan Zhang, Dian Zheng, Ziyu Guo, Yimeng Jia, Kaituo Feng, Hao Yu, Yexin Liu, Yan Feng, Peng Pei, Xunliang Cai, Linjiang Huang, Hongsheng Li, Si Liu

**发布时间:** 2025-12-05

**备注:** Project page: https://appletea233.github.io/think-while-edit

### GPT解析

### 总结

本文提出了一种名为EditThinker的深思熟虑型图像编辑框架，通过模拟人类认知循环的Think-while-Edit周期，显著提高了图像编辑模型的指令遵循能力。

### 背景

基于指令的图像编辑已成为一个突出的研究领域，借助图像生成基础模型已实现高美学质量，但指令遵循能力仍是主要挑战。

### 目的

解决现有方法因内在随机性和缺乏深思熟虑而导致的单次编辑成功率有限的问题，提出一种能让模型在编辑过程中'思考'的框架。

### 方法

开发一个Think-while-Edit循环：批判结果、细化指令、重复生成直至满意；训练单一多语言大模型EditThinker作为推理引擎，共同生成批判分数、推理过程和细化指令；使用强化学习使模型思考与编辑保持一致，生成更有针对性的指令改进。

### 主要发现

在四个基准测试上的广泛实验表明，该方法显著提高了任何图像编辑模型的指令遵循能力，提高幅度较大。

### 结论

将发布数据构建框架、数据集和模型以造福社区，促进基于指令的图像编辑研究发展。

### 翻译

基于指令的图像编辑已成为一个突出的研究领域，它受益于图像生成基础模型，已实现高美学质量，使得指令遵循能力成为主要挑战。现有方法通过监督学习或强化学习提高指令遵循度，但由于内在随机性和缺乏深思熟虑，单次成功率仍然有限。在这项工作中，我们提出了一个深思熟虑的编辑框架，让模型在编辑时'思考'，它通过迭代执行一个思考-编辑循环来模拟人类认知循环：批判结果并细化指令，然后重复生成直至满意。具体来说，我们训练了一个单一的多语言大模型EditThinker，作为该框架的推理引擎，共同生成批判分数、推理过程和细化指令。我们采用强化学习使EditThinker的思考与编辑保持一致，从而生成更有针对性的指令改进。在四个基准测试上的广泛实验表明，我们的方法显著提高了任何图像编辑模型的指令遵循能力。我们将发布数据构建框架、数据集和模型以造福社区。


### 论文摘要

Instruction-based image editing has emerged as a prominent research area, which, benefiting from image generation foundation models, have achieved high aesthetic quality, making instruction-following capability the primary challenge. Existing approaches improve instruction adherence via supervised or reinforcement learning, yet single-turn success rates remain limited due to inherent stochasticity and a lack of deliberation. In this work, we propose a deliberative editing framework to 'think' while they edit, which simulates the human cognitive loop by iteratively executing a Think-while-Edit cycle: Critiquing results and Refining instructions , followed by Repeating the generation until satisfactory. Specifically, we train a single MLLM, EditThinker, to act as the reasoning engine of this framework, which jointly produce the critique score, reasoning process, and refined instructions. We employ reinforcement learning to align the EditThinker's thinking with its editing, thereby generating more targeted instruction improvements. Extensive experiments on four benchmarks demonstrate that our approach significantly improves the instruction-following capability of any image editing model by a large margin. We will release our data construction framework, datasets, and models to benefit the community.

---

## 51. M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG

**论文链接:** [http://arxiv.org/abs/2512.05959v1](http://arxiv.org/abs/2512.05959v1)

**作者:** David Anugraha, Patrick Amadeus Irawan, Anshul Singh, En-Shiun Annie Lee, Genta Indra Winata

**发布时间:** 2025-12-05

**备注:** Preprint

### GPT解析

### 总结

该研究引入了M4-RAG，一个覆盖42种语言和56种区域方言的大规模基准测试，用于评估多语言多模态检索增强视觉问答系统。

### 背景

视觉语言模型在视觉问答任务中表现出色，但受限于静态训练数据。检索增强生成可以访问最新、文化相关和多语言信息，但多模态多语言RAG领域探索不足。

### 目的

创建M4-RAG基准测试，包含超过80,000种文化多样的图像-问题对，用于评估跨语言和模态的检索增强视觉问答能力。

### 方法

构建受控检索环境，包含数百万个与查询领域相关的精心策划的多语言文档，模拟真实检索条件同时确保实验一致性。

### 主要发现

RAG持续有利于较小的视觉语言模型，但无法扩展到更大模型，甚至经常降低其性能，揭示了模型大小与当前检索有效性之间的关键不匹配。

### 结论

M4-RAG为推进能够跨语言、模态和文化背景无缝推理的下一代RAG系统提供了基础。

### 翻译

该论文摘要的中文翻译已包含在上述各点中，无需额外翻译。


### 论文摘要

Vision-language models (VLMs) have achieved strong performance in visual question answering (VQA), yet they remain constrained by static training data. Retrieval-Augmented Generation (RAG) mitigates this limitation by enabling access to up-to-date, culturally grounded, and multilingual information; however, multilingual multimodal RAG remains largely underexplored. We introduce M4-RAG, a massive-scale benchmark covering 42 languages and 56 regional dialects and registers, comprising over 80,000 culturally diverse image-question pairs for evaluating retrieval-augmented VQA across languages and modalities. To balance realism with reproducibility, we build a controlled retrieval environment containing millions of carefully curated multilingual documents relevant to the query domains, approximating real-world retrieval conditions while ensuring consistent experimentation. Our systematic evaluation reveals that although RAG consistently benefits smaller VLMs, it fails to scale to larger models and often even degrades their performance, exposing a critical mismatch between model size and current retrieval effectiveness. M4-RAG provides a foundation for advancing next-generation RAG systems capable of reasoning seamlessly across languages, modalities, and cultural contexts.

---

## 52. SymPyBench: A Dynamic Benchmark for Scientific Reasoning with Executable Python Code

**论文链接:** [http://arxiv.org/abs/2512.05954v1](http://arxiv.org/abs/2512.05954v1)

**作者:** Shima Imani, Seungwhan Moon, Adel Ahmadyan, Lu Zhang, Kirmani Ahmed, Babak Damavandi

**发布时间:** 2025-12-05

### GPT解析

### 总结

研究团队开发了一个名为SymPyBench的大规模物理问题基准测试，包含15,045个大学级别的物理问题，具有完全参数化的特性，并引入了新的评估指标来测试语言模型的科学推理能力。

### 背景

当前缺乏一个全面且灵活的基准测试来评估语言模型在物理问题解决方面的能力。现有基准测试往往规模有限，且缺乏动态性和多样性。

### 目的

创建一个大规模、多样化的物理问题基准测试，以评估和改进语言模型在科学推理方面的能力，并引入新的评估指标来更全面地衡量模型性能。

### 方法

构建了一个包含15,045个大学级别物理问题的合成基准测试，每个问题都完全参数化并配有结构化推理和Python代码。基准测试包含三种问题类型：MC-Symbolic、MC-Numerical和自由形式回答。此外，引入了三个新的评估指标：一致性分数、失败率和混淆率。

### 主要发现

通过对最先进的指令调整语言模型进行实验，研究发现这些模型在科学推理方面既有优势也有局限性。新引入的评估指标能够更全面地量化模型在不同问题变体上的表现。

### 结论

SymPyBench基准测试为开发更强大和可解释的推理系统提供了基础，通过其动态、代码驱动的特性，能够更全面地评估和改进语言模型在科学推理方面的能力。

### 翻译

我们引入了一个大规模合成基准测试，包含15,045个大学级别的物理问题（90/10%的训练/测试分割）。每个问题都是完全参数化的，支持无限范围的输入配置，并附有结构化的逐步推理和可执行的Python代码，可为任何参数集生成真实解决方案。该基准测试包含三种问题类型：MC-Symbolic（具有符号选项的多项选择）、MC-Numerical（具有数值选项的多项选择）和自由形式（开放式回答）。这些多样化的格式测试互补的推理技能。通过利用基准测试的动态、代码驱动特性，我们在标准准确性之外引入了三个新的评估指标：一致性分数、失败率和混淆率，这些指标量化了问题变体之间的变化性和不确定性。使用最先进的指令调整语言模型进行的实验揭示了科学推理的优势和局限性，将SymPyBench定位为开发更强大和可解释的推理系统的基础。


### 论文摘要

We introduce, a large-scale synthetic benchmark of 15,045 university-level physics problems (90/10% train/test split). Each problem is fully parameterized, supporting an effectively infinite range of input configurations, and is accompanied by structured, step-by-step reasoning and executable Python code that produces the ground-truth solution for any parameter set. The benchmark contains three question types: MC-Symbolic (multiple-choice with symbolic options), MC-Numerical (multiple-choice with numerical options), and free-form (open-ended responses). These diverse formats test complementary reasoning skills. By leveraging the dynamic, code-driven nature of the benchmark, we introduce three novel evaluation metrics in addition to standard accuracy: Consistency Score, Failure Rate, and Confusion Rate, that quantify variability and uncertainty across problem variants. Experiments with state-of-the-art instruction-tuned language models reveal both strengths and limitations in scientific reasoning, positioning SymPyBench as a foundation for developing more robust and interpretable reasoning systems

---

## 53. TRACE: A Framework for Analyzing and Enhancing Stepwise Reasoning in Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2512.05943v1](http://arxiv.org/abs/2512.05943v1)

**作者:** Shima Imani, Seungwhan Moon, Lambert Mathias, Lu Zhang, Babak Damavandi

**发布时间:** 2025-12-05

### GPT解析

### 总结

TRACE是一个透明的推理和一致性评估框架，通过辅助推理集分解复杂问题，评估中间步骤，并暴露标准评估忽略的失败，有助于识别推理错误的具体位置，为模型改进提供指导。

### 背景

大型视觉语言模型在数学和科学推理方面仍然是一个开放性挑战，标准的最终答案评估往往会掩盖推理错误，导致静默失败持续存在。

### 目的

解决上述问题，引入TRACE框架，用于诊断推理轨迹而不仅仅是最终结果。

### 方法

TRACE的核心是利用辅助推理集（紧凑的子问题-答案对），能够分解复杂问题，通过基于一致性的指标评估中间步骤，暴露标准评估所忽略的失败。

### 主要发现

实验表明，辅助推理集之间的一致性与最终答案的正确性相关，有助于识别推理失败出现的具体步骤，为模型改进提供可操作的信号；TRACE还定义了置信区域，区分可靠和不可靠的推理路径，支持有效的过滤、调试和模型改进。

### 结论

TRACE框架提供了一种更全面的方法来评估和改进视觉语言模型的推理能力，通过关注推理过程而非仅仅最终结果，可以更准确地识别和解决模型中的问题。

### 翻译

可靠数学和科学推理对大型视觉语言模型来说仍然是一个开放性挑战。标准的最终答案评估往往会掩盖推理错误，导致静默失败持续存在。为解决这一差距，我们引入了TRACE，一个用于透明推理和一致性评估的框架，它诊断推理轨迹而不仅仅是最终结果。其核心是TRACE利用辅助推理集，这些是紧凑的子问题-答案对，能够分解复杂问题，通过基于一致性的指标评估中间步骤，并暴露标准评估所忽略的失败。我们的实验表明，辅助推理集之间的一致性与最终答案的正确性相关，有助于识别推理失败出现的具体步骤，为模型改进提供可操作的信号。此外，TRACE定义了置信区域，区分可靠和不可靠的推理路径，支持有效的过滤、调试和模型改进。


### 论文摘要

Reliable mathematical and scientific reasoning remains an open challenge for large vision-language models. Standard final-answer evaluation often masks reasoning errors, allowing silent failures to persist. To address this gap, we introduce TRACE, a framework for Transparent Reasoning And Consistency Evaluation that diagnoses reasoning trajectories rather than only end results. At its core, TRACE leverages Auxiliary Reasoning Sets, compact sub question answer pairs that decompose complex problems, evaluate intermediate steps through consistency-based metrics, and expose failures overlooked by standard evaluation. Our experiments show that consistency across ARS correlates with final-answer correctness and helps pinpoint the reasoning steps where failures arise, offering actionable signals for model improvement. Furthermore, TRACE defines confidence regions that distinguish reliable from unreliable reasoning paths, supporting effective filtering, debugging, and model refinement.

---

## 54. Zoom in, Click out: Unlocking and Evaluating the Potential of Zooming for GUI Grounding

**论文链接:** [http://arxiv.org/abs/2512.05941v1](http://arxiv.org/abs/2512.05941v1)

**作者:** Zhiyuan Jiang, Shenghao Xie, Wenyi Li, Wenqiang Zu, Peihang Li, Jiahao Qiu, Siqi Pei, Lei Ma, Tiejun Huang, Mengdi Wang, Shilong Liu

**发布时间:** 2025-12-05

**备注:** Code is available at https://github.com/Princeton-AI2-Lab/ZoomClick

### GPT解析

### 总结

该论文提出了一种基于缩放先验的无需训练方法ZoomClick，用于解决GUI grounding中的挑战，通过表征缩放的四个关键特性实现动态空间聚焦和自适应上下文切换，在多个基准测试上取得了最先进的结果。

### 背景

Grounding是构建图形用户界面(GUI)代理的基本能力，现有方法依赖于大规模边界框监督，但仍面临跨平台泛化、复杂布局分析和细粒度元素定位等挑战。

### 目的

研究缩放(zoom)作为GUI grounding中强大但未被充分探索的先验，提出一种无需训练的方法ZoomClick。

### 方法

通过表征缩放的四个关键特性（预缩放、深度、缩小尺寸、最小裁剪尺寸），实现动态空间聚焦和自适应上下文切换。

### 主要发现

实验证明该方法显著提高了通用视觉语言模型和专业化GUI grounding模型的性能，UI-Venus-72B在ScreenSpot-Pro上达到73.1%的成功率，提出了GUIZoom-Bench基准用于评估模型对缩放的适应性。

### 结论

缩放作为一种先验可以提升GUI grounding性能，提出的基准有助于未来研究改进缩放，进一步扩展训练和测试时的应用。

### 翻译

Grounding是构建图形用户界面(GUI)代理的基本能力。尽管现有方法依赖于大规模边界框监督，它们仍然面临各种挑战，如跨平台泛化、复杂布局分析和细粒度元素定位。在本文中，我们研究了缩放作为GUI grounding中强大但未被充分探索的先验，并提出了一种无需训练的方法ZoomClick。通过表征缩放的四个关键特性（即预缩放、深度、缩小尺寸、最小裁剪尺寸），我们释放了其在动态空间聚焦和自适应上下文切换方面的全部能力。实验证明，我们的方法显著提高了通用视觉语言模型和专业化GUI grounding模型的性能，在几个主流基准测试上取得了最先进的结果；例如，UI-Venus-72B在ScreenSpot-Pro上达到73.1%的成功率。此外，我们提出了GUIZoom-Bench基准，用于评估模型对缩放的适应性，旨在激励未来研究改进缩放，以进一步扩展GUI grounding任务中的训练和测试时应用。


### 论文摘要

Grounding is a fundamental capability for building graphical user interface (GUI) agents. Although existing approaches rely on large-scale bounding box supervision, they still face various challenges, such as cross-platform generalization, complex layout analysis, and fine-grained element localization. In this paper, we investigate zoom as a strong yet underexplored prior for GUI grounding, and propose a training-free method, ZoomClick. By characterizing four key properties of zoom (i.e., pre-zoom, depth, shrink size, minimal crop size), we unlock its full capabilities for dynamic spatial focusing and adaptive context switching. Experiments demonstrate that our method significantly boosts the performance of both general vision-language and specialized GUI grounding models, achieving state-of-the-art results on several mainstream benchmarks; for example, UI-Venus-72B attains a 73.1% success rate on ScreenSpot-Pro. Furthermore, we present GUIZoom-Bench, a benchmark for evaluating model adaptability to zoom, aiming to inspire future research on improving zoom for further training and test-time scaling in GUI grounding tasks.

---

## 55. A Comparative Study on Synthetic Facial Data Generation Techniques for Face Recognition

**论文链接:** [http://arxiv.org/abs/2512.05928v1](http://arxiv.org/abs/2512.05928v1)

**作者:** Pedro Vidal, Bernardo Biesseck, Luiz E. L. Coelho, Roger Granada, David Menotti

**发布时间:** 2025-12-05

**备注:** 18 pages, 17 figures

### GPT解析

### 总结

这项研究比较了不同技术生成的合成面部数据集在人脸识别任务中的有效性，结果表明合成数据能够捕捉真实的变异性，但需要进一步研究以缩小与真实数据的性能差距。

### 背景

人脸识别已成为广泛使用的身份验证和识别方法，应用于安全访问和寻找失踪人员。其成功主要归功于深度学习，利用大型数据集和有效损失函数学习判别性特征。尽管如此，人脸识别仍面临可解释性、人口统计偏见、隐私以及对抗衰老、姿态变化、光照变化、遮挡和面部表情的鲁棒性等挑战。隐私法规也导致多个数据集质量下降，引发法律、伦理和隐私问题。

### 目的

比较不同技术生成的合成面部数据集在人脸识别任务中的有效性。

### 方法

在八个领先数据集上评估识别准确率、前一名和前五名的识别率以及在假阳性率为0.01%时的真阳性率，提供文献中未广泛探索的比较分析。

### 主要发现

合成数据能够捕捉真实的变异性，扩散模型、生成对抗网络和三维模型等技术显示出实质性进展，但仍存在挑战。需要进一步研究以缩小与真实数据的性能差距。

### 结论

合成面部数据生成是一个有前景的解决方案，它可以减轻隐私问题， enable对受控面部属性的实验，减轻人口统计偏见，并提供补充数据来改进在真实数据上训练的模型。

### 翻译

人脸识别已成为广泛使用的身份验证和识别方法，应用于安全访问和寻找失踪人员。其成功主要归功于深度学习，利用大型数据集和有效损失函数学习判别性特征。尽管有这些进展，人脸识别仍面临可解释性、人口统计偏见、隐私以及对抗衰老、姿态变化、光照变化、遮挡和面部表情的鲁棒性等挑战。隐私法规也导致多个数据集质量下降，引发法律、伦理和隐私问题。合成面部数据生成已被提出作为一种有前景的解决方案。它可以减轻隐私问题， enable对受控面部属性的实验，减轻人口统计偏见，并提供补充数据来改进在真实数据上训练的模型。本研究比较了不同技术生成的合成面部数据集在人脸识别任务中的有效性。我们在八个领先数据集上评估了准确性、rank-1、rank-5以及在假阳性率为0.01%时的真阳性率，提供了文献中未广泛探索的比较分析。结果表明合成数据能够捕捉真实的变异性，同时强调需要进一步研究以缩小与真实数据的性能差距。扩散模型、GANs和3D模型等技术显示出实质性进展；然而，挑战仍然存在。


### 论文摘要

Facial recognition has become a widely used method for authentication and identification, with applications for secure access and locating missing persons. Its success is largely attributed to deep learning, which leverages large datasets and effective loss functions to learn discriminative features. Despite these advances, facial recognition still faces challenges in explainability, demographic bias, privacy, and robustness to aging, pose variations, lighting changes, occlusions, and facial expressions. Privacy regulations have also led to the degradation of several datasets, raising legal, ethical, and privacy concerns. Synthetic facial data generation has been proposed as a promising solution. It mitigates privacy issues, enables experimentation with controlled facial attributes, alleviates demographic bias, and provides supplementary data to improve models trained on real data. This study compares the effectiveness of synthetic facial datasets generated using different techniques in facial recognition tasks. We evaluate accuracy, rank-1, rank-5, and the true positive rate at a false positive rate of 0.01% on eight leading datasets, offering a comparative analysis not extensively explored in the literature. Results demonstrate the ability of synthetic data to capture realistic variations while emphasizing the need for further research to close the performance gap with real data. Techniques such as diffusion models, GANs, and 3D models show substantial progress; however, challenges remain.

---

## 56. The Bayesian Way: Uncertainty, Learning, and Statistical Reasoning

**论文链接:** [http://arxiv.org/abs/2512.05883v1](http://arxiv.org/abs/2512.05883v1)

**作者:** Juan Sosa, Carlos A. Martínez, Danna Cruz

**发布时间:** 2025-12-05

**备注:** 56 pages, 1 table, 0 figures

### GPT解析

### 总结

这篇论文全面介绍了贝叶斯推断，结合了历史背景、理论基础和核心分析示例。

### 背景

从贝叶斯定理开始，阐述了贝叶斯与频率学派方法之间的哲学区别，并发展了估计、区间构建、假设检验和预测的推断框架。

### 目的

为学生和研究人员提供一个严谨且易于理解的入门点，以便他们在统计实践中采用贝叶斯视角。

### 方法

通过经典模型说明如何形式化整合先验信息和观测数据以获得后验分布；探讨了损失函数、可信区间、贝叶斯因子、可识别性和渐近行为等关键概念；强调了在经典设置中的解析可处理性，并概述了依赖于模拟方法的现代扩展。

### 主要发现

展示了如何将先验信息和观测数据形式化整合以产生后验分布；讨论了与先验规范和模型评估相关的挑战。

### 结论

尽管专注于基础概念，但本文为将贝叶斯方法应用于当代领域（如分层建模、非参数方法以及时间序列、空间数据、网络和政治科学中的结构化应用）奠定了基础。

### 翻译

本文对贝叶斯推断进行了全面介绍，结合了历史背景、理论基础和核心分析示例。从贝叶斯定理以及贝叶斯与频率学派方法之间的哲学区别开始，我们构建了用于估计、区间构建、假设检验和预测的推断框架。通过经典模型，我们说明了如何形式化整合先验信息和观测数据以产生后验分布。我们还探讨了关键概念，包括损失函数、可信区间、贝叶斯因子、可识别性和渐近行为。在强调经典设置中的解析可处理性的同时，我们概述了依赖于模拟方法的现代扩展，并讨论了与先验规范和模型评估相关的挑战。虽然专注于基础概念，但本文为将贝叶斯方法应用于当代领域（如分层建模、非参数方法以及时间序列、空间数据、网络和政治科学中的结构化应用）奠定了基础。其目标是为寻求在统计实践中采用贝叶斯视角的学生和研究人员提供一个严谨且易于理解的入门点。


### 论文摘要

This paper offers a comprehensive introduction to Bayesian inference, combining historical context, theoretical foundations, and core analytical examples. Beginning with Bayes' theorem and the philosophical distinctions between Bayesian and frequentist approaches, we develop the inferential framework for estimation, interval construction, hypothesis testing, and prediction. Through canonical models, we illustrate how prior information and observed data are formally integrated to yield posterior distributions. We also explore key concepts including loss functions, credible intervals, Bayes factors, identifiability, and asymptotic behavior. While emphasizing analytical tractability in classical settings, we outline modern extensions that rely on simulation-based methods and discuss challenges related to prior specification and model evaluation. Though focused on foundational ideas, this paper sets the stage for applying Bayesian methods in contemporary domains such as hierarchical modeling, nonparametrics, and structured applications in time series, spatial data, networks, and political science. The goal is to provide a rigorous yet accessible entry point for students and researchers seeking to adopt a Bayesian perspective in statistical practice.

---

## 57. Neural Coherence : Find higher performance to out-of-distribution tasks from few samples

**论文链接:** [http://arxiv.org/abs/2512.05880v1](http://arxiv.org/abs/2512.05880v1)

**作者:** Simon Guiroy, Mats Richter, Sarath Chandar, Christopher Pal

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种名为神经一致性的新方法，用于在数据稀缺、未标记且分布外的情况下进行模型选择，相比传统方法显著提高了模型在下游任务上的泛化能力。

### 背景

微调预训练的大视觉模型已成为创建先进下游任务模型的常见做法，但从大型训练运行中的众多模型检查点中确定最佳起点仍是一个开放性问题。当目标任务数据稀缺、未标记且分布外时，依赖分布内验证数据的常见方法变得不可靠或不适用。

### 目的

开发一种仅使用目标任务中的少量未标记示例就能可靠运行的模型选择新方法。

### 方法

基于神经一致性概念，通过表征模型在源域和目标域的激活统计特征，定义高数据效率的模型选择方法。

### 主要发现

与既定基线相比，该方法在Food-101、PlantNet-300K和iNaturalist等不同目标域上显著提高了泛化能力，且在训练数据选择中也表现出有效性。

### 结论

神经一致性方法在数据受限场景下能有效进行模型选择，提高模型泛化能力，具有广泛的应用价值。

### 翻译

为了创建许多下游任务的最先进模型，微调预训练的大型视觉模型已成为常见做法。然而，如何从大型训练运行中的众多可能模型检查点中确定最佳起点仍是一个开放性问题。当目标任务的数据稀缺、未标记且分布外时，这变得尤为重要。在这种情况下，依赖于分布内验证数据的常见方法变得不可靠或不适用。本文提出了一种仅使用目标任务中的少量未标记示例就能可靠运行的模型选择新方法。我们的方法基于一个新概念：神经一致性，它涉及对源域和目标域的模型激活统计特征进行表征，从而能够定义高数据效率的模型选择方法。我们提供了在ImageNet1K上预训练模型并在包含Food-101、PlantNet-300K和iNaturalist的目标域中进行实验。我们还在许多元学习设置中对其进行了评估。与既定基线相比，我们的方法在这些不同目标域上显著提高了泛化能力。此外，通过展示其在训练数据选择中的有效性，我们进一步证明了神经一致性作为一种强大原理的多功能性。


### 论文摘要

To create state-of-the-art models for many downstream tasks, it has become common practice to fine-tune a pre-trained large vision model. However, it remains an open question of how to best determine which of the many possible model checkpoints resulting from a large training run to use as the starting point. This becomes especially important when data for the target task of interest is scarce, unlabeled and out-of-distribution. In such scenarios, common methods relying on in-distribution validation data become unreliable or inapplicable. This work proposes a novel approach for model selection that operates reliably on just a few unlabeled examples from the target task. Our approach is based on a novel concept: Neural Coherence, which entails characterizing a model's activation statistics for source and target domains, allowing one to define model selection methods with high data-efficiency. We provide experiments where models are pre-trained on ImageNet1K and examine target domains consisting of Food-101, PlantNet-300K and iNaturalist. We also evaluate it in many meta-learning settings. Our approach significantly improves generalization across these different target domains compared to established baselines. We further demonstrate the versatility of Neural Coherence as a powerful principle by showing its effectiveness in training data selection.

---

## 58. Sparse Attention Post-Training for Mechanistic Interpretability

**论文链接:** [http://arxiv.org/abs/2512.05865v1](http://arxiv.org/abs/2512.05865v1)

**作者:** Florent Draye, Anson Lei, Ingmar Posner, Bernhard Schölkopf

**发布时间:** 2025-12-05

### GPT解析

### 总结

这篇论文介绍了一种简单的事后训练方法，可以在不牺牲性能的情况下使transformer注意力变得稀疏。通过稀疏正则化，可以在保持原始预训练损失的同时将注意力连接减少到约0.3%，并揭示出更有组织和可解释的连接模式。

### 背景

transformer模型在自然语言处理等领域广泛应用，但其注意力机制通常是密集的，导致计算复杂且难以解释。现有稀疏注意力方法主要关注计算效率，而非模型结构和可解释性。

### 目的

开发一种简单的事后训练方法，使transformer注意力变得稀疏而不牺牲性能，同时探索稀疏性作为结构先验的潜力，揭示模型内部更组织和可解释的连接模式。

### 方法

在约束损失目标下应用灵活的稀疏正则化，对高达10亿参数的transformer模型进行训练，将稀疏性视为结构先验而非仅为了计算效率。

### 主要发现

1) 可以保持原始预训练损失的同时将注意力连接减少到约0.3%；2) 稀疏性暴露出更有组织和可解释的连接模式；3) 局部稀疏性会级联到全局电路简化，特定任务电路涉及的组件和连接边大幅减少；4) transformer注意力的大部分计算可能是冗余的。

### 结论

稀疏性可以作为构建更结构化和可解释transformer模型的指导原则，通过简单的事后训练方法可以在不牺牲性能的情况下显著减少注意力连接，揭示模型内部更简洁的电路结构。

### 翻译

我们介绍了一种简单的事后训练方法，可以在不牺牲性能的情况下使transformer注意力变得稀疏。在约束损失目标下应用灵活的稀疏正则化，我们在高达10亿参数的模型上证明，可以在保持原始预训练损失的同时，将注意力连接减少到约0.3%。与为计算效率而设计的稀疏注意力方法不同，我们的方法利用稀疏性作为结构先验：它保留了能力，同时暴露出更有组织和可解释的连接模式。我们发现这种局部稀疏性会级联到全局电路简化：特定任务的电路涉及更少的组件（注意力头和MLP），连接它们的边最多减少100倍。这些结果表明，transformer注意力可以变得稀疏几个数量级，表明其大部分计算是冗余的，并且稀疏性可以作为构建更结构化和可解释模型的指导原则。


### 论文摘要

We introduce a simple post-training method that makes transformer attention sparse without sacrificing performance. Applying a flexible sparsity regularisation under a constrained-loss objective, we show on models up to 1B parameters that it is possible to retain the original pretraining loss while reducing attention connectivity to $\approx 0.3 \%$ of its edges. Unlike sparse-attention methods designed for computational efficiency, our approach leverages sparsity as a structural prior: it preserves capability while exposing a more organized and interpretable connectivity pattern. We find that this local sparsity cascades into global circuit simplification: task-specific circuits involve far fewer components (attention heads and MLPs) with up to 100x fewer edges connecting them. These results demonstrate that transformer attention can be made orders of magnitude sparser, suggesting that much of its computation is redundant and that sparsity may serve as a guiding principle for more structured and interpretable models.

---

## 59. VRSA: Jailbreaking Multimodal Large Language Models through Visual Reasoning Sequential Attack

**论文链接:** [http://arxiv.org/abs/2512.05853v1](http://arxiv.org/abs/2512.05853v1)

**作者:** Shiji Zhao, Shukun Xiong, Yao Huang, Yan Jin, Zhenyu Wu, Jiyang Guan, Ranjie Duan, Jialing Tao, Hui Xue, Xingxing Wei

**发布时间:** 2025-12-05

### GPT解析

### 总结

该研究提出了一种名为视觉推理序列攻击(VRSA)的新型越狱攻击方法，针对多模态大语言模型在视觉模态中的安全风险进行了全面评估。

### 背景

多模态大语言模型(MLLMs)因其跨模态理解和生成能力被广泛应用，但多模态特性也使其更容易受到越狱攻击。现有研究主要关注文本模态的安全风险，而忽视了视觉模态中的类似威胁。

### 目的

全面评估视觉推理任务中多模态大语言模型的潜在安全风险，并提出针对性的攻击方法。

### 方法

提出视觉推理序列攻击(VRSA)，通过将有害文本分解为 sequentially 相关的子图像来诱导模型输出有害内容；引入自适应场景优化增强图像序列合理性；采用语义连贯完成确保生成图像的语义连贯性；利用文本-图像一致性对齐保持语义一致性。

### 主要发现

VRSA方法在开源和闭源MLLMs(如GPT-4o和Claude-4.5-Sonnet)上均实现了比现有最先进越狱攻击方法更高的攻击成功率，表明视觉模态存在显著的安全风险。

### 结论

视觉模态是多模态大语言模型安全防御中不可忽视的重要方面，需要开发针对性的防御策略来应对视觉推理序列攻击等新型威胁。

### 翻译

多模态大语言模型(MLLMs)因其强大的跨模态理解和生成能力而被广泛应用于各个领域。然而，更多的模态也带来了更多的漏洞，可能被用于越狱攻击，导致MLLMs输出有害内容。由于MLLMs的强大推理能力，之前的越狱攻击主要探索文本模态中的推理安全风险，而视觉模态中的类似威胁在很大程度上被忽视。为了全面评估视觉推理任务中的潜在安全风险，我们提出了视觉推理序列攻击(VRSA)，通过将原始有害文本分解为几个 sequentially 相关的子图像，诱导MLLMs逐步外化和聚合完整的有害意图。特别是，为了增强图像序列中场景的合理性，我们提出了自适应场景优化来优化与原始有害查询最相关的场景。为确保生成图像的语义连贯性，我们提出了语义连贯完成，结合场景中的上下文信息迭代重写每个子文本。此外，我们还提出了文本-图像一致性对齐以保持语义一致性。一系列实验表明，与最先进的越狱攻击方法相比，VRSA在开源和闭源MLLMs(如GPT-4o和Claude-4.5-Sonnet)上都能实现更高的攻击成功率。


### 论文摘要

Multimodal Large Language Models (MLLMs) are widely used in various fields due to their powerful cross-modal comprehension and generation capabilities. However, more modalities bring more vulnerabilities to being utilized for jailbreak attacks, which induces MLLMs to output harmful content. Due to the strong reasoning ability of MLLMs, previous jailbreak attacks try to explore reasoning safety risk in text modal, while similar threats have been largely overlooked in the visual modal. To fully evaluate potential safety risks in the visual reasoning task, we propose Visual Reasoning Sequential Attack (VRSA), which induces MLLMs to gradually externalize and aggregate complete harmful intent by decomposing the original harmful text into several sequentially related sub-images. In particular, to enhance the rationality of the scene in the image sequence, we propose Adaptive Scene Refinement to optimize the scene most relevant to the original harmful query. To ensure the semantic continuity of the generated image, we propose Semantic Coherent Completion to iteratively rewrite each sub-text combined with contextual information in this scene. In addition, we propose Text-Image Consistency Alignment to keep the semantical consistency. A series of experiments demonstrates that the VRSA can achieve a higher attack success rate compared with the state-of-the-art jailbreak attack methods on both the open-source and closed-source MLLMs such as GPT-4o and Claude-4.5-Sonnet.

---

## 60. NEAT: Neighborhood-Guided, Efficient, Autoregressive Set Transformer for 3D Molecular Generation

**论文链接:** [http://arxiv.org/abs/2512.05844v1](http://arxiv.org/abs/2512.05844v1)

**作者:** Daniel Rose, Roxane Axel Jacob, Johannes Kirchmair, Thierry Langer

**发布时间:** 2025-12-05

### GPT解析

### 总结

NEAT是一种新的自回归模型，用于3D分子结构生成，解决了原子排列不变性问题，实现了高效和准确的分子设计。

### 背景

自回归模型是3D分子结构生成的有前途的替代方法，但面临标记顺序假设的限制，因为分子图中的下一个标记预测应该对原子排列不变。

### 目的

解决自回归模型在分子结构生成中的顺序假设问题，实现原子排列不变的高效分子生成。

### 方法

引入NEAT（Neighborhood-guided, Efficient, Autoregressive, Set Transformer），将分子图视为原子集合，使用自回归流模型学习图边界上可接受标记的顺序无关分布。

### 主要发现

NEAT在3D分子生成中达到最先进性能，具有高计算效率和原子级排列不变性。

### 结论

NEAT为可扩展的分子设计建立了实用基础。

### 翻译

自回归模型是3D分子结构生成的有前途的替代方法，但一个关键限制是标记顺序的假设：虽然文本有自然的顺序，但给定分子图前缀的下一个标记预测应该对原子排列不变。先前工作通过使用规范顺序或焦点原子来回避这种不匹配。我们认为这是不必要的。我们引入NEAT，一种邻域引导的、高效的、自回归的集合变换器，它将分子图视为原子集合，并使用自回归流模型学习图边界上可接受标记的顺序无关分布。NEAT在3D分子生成中达到最先进性能，具有高计算效率和原子级排列不变性，为可扩展的分子设计建立了实用基础。


### 论文摘要

Autoregressive models are a promising alternative to diffusion-based models for 3D molecular structure generation. However, a key limitation is the assumption of a token order: while text has a natural sequential order, the next token prediction given a molecular graph prefix should be invariant to atom permutations. Previous works sidestepped this mismatch by using canonical orders or focus atoms. We argue that this is unnecessary. We introduce NEAT, a Neighborhood-guided, Efficient, Autoregressive, Set Transformer that treats molecular graphs as sets of atoms and learns the order-agnostic distribution over admissible tokens at the graph boundary with an autoregressive flow model. NEAT approaches state-of-the-art performance in 3D molecular generation with high computational efficiency and atom-level permutation invariance, establishing a practical foundation for scalable molecular design.

---

## 61. Invariant Price of Anarchy: a Metric for Welfarist Traffic Control

**论文链接:** [http://arxiv.org/abs/2512.05843v1](http://arxiv.org/abs/2512.05843v1)

**作者:** Ilia Shilov, Mingjia He, Heinrich H. Nax, Emilio Frazzoli, Gioele Zardini, Saverio Bolognani

**发布时间:** 2025-12-05

### GPT解析

### 总结

本研究提出了一种不变PoA(无政府状态价格)指标，解决了传统PoA分析中成本定义的任意缩放和平移问题，通过社会选择理论建立了新的效率评估框架。

### 背景

传统PoA分析依赖于精确的数值成本，但在实际应用中，成本代表参与者偏好，可能仅定义为任意缩放和平移，这会导致相同策略下产生不同的效率估计。

### 目的

解决传统PoA分析中因成本定义的任意性导致的效率评估不一致问题，建立稳健有效的效率评估框架以指导大规模基础设施设计政策。

### 方法

从社会选择理论出发，定义不变PoA指标，将可接受的成本转换与参与者成本的比较程度相关联，推导出确保效率评估不依赖于个人成本任意重新缩放或平移的社会福利函数。

### 主要发现

案例研究表明，相同的收费策略可能导致显著不同的效率估计，这取决于对成本比较性的假设；明确的公理化基础对于定义效率指标和指导政策制定是必要的。

### 结论

不变PoA框架为大规模基础设施设计中的政策制定提供了稳健有效的效率评估基础，强调了明确公理化基础的重要性。

### 翻译

无政府状态价格(PoA)是量化社会技术系统中无效率的标准指标，广泛用于指导交通收费等政策。传统PoA分析依赖于精确的数值成本。然而，在许多情况下，成本代表参与者的偏好，可能仅定义为任意缩放和平移，反映了信息和建模的不确定性。我们观察到，虽然这种转换保持均衡和最优结果不变，但会改变PoA值。为解决此问题，我们依赖社会选择理论的结果，定义了不变PoA。通过将可接受的转换与参与者成本的比较程度相关联，我们推导出确保效率评估不依赖于个人成本的任意重新缩放或平移的特定社会福利函数。在玩具示例和苏黎世网络上的案例研究表明，相同的收费策略可能导致显著不同的效率估计，这取决于假设的比较性。我们的框架因此表明，明确的公理化基础对于定义效率指标以及稳健有效地指导大规模基础设施设计中的政策是必要的。


### 论文摘要

The Price of Anarchy (PoA) is a standard metric for quantifying inefficiency in socio-technical systems, widely used to guide policies like traffic tolling. Conventional PoA analysis relies on exact numerical costs. However, in many settings, costs represent agents' preferences and may be defined only up to possibly arbitrary scaling and shifting, representing informational and modeling ambiguities. We observe that while such transformations preserve equilibrium and optimal outcomes, they change the PoA value. To resolve this issue, we rely on results from Social Choice Theory and define the Invariant PoA. By connecting admissible transformations to degrees of comparability of agents' costs, we derive the specific social welfare functions which ensure that efficiency evaluations do not depend on arbitrary rescalings or translations of individual costs. Case studies on a toy example and the Zurich network demonstrate that identical tolling strategies can lead to substantially different efficiency estimates depending on the assumed comparability. Our framework thus demonstrates that explicit axiomatic foundations are necessary in order to define efficiency metrics and to appropriately guide policy in large-scale infrastructure design robustly and effectively.

---

## 62. Multimodal Oncology Agent for IDH1 Mutation Prediction in Low-Grade Glioma

**论文链接:** [http://arxiv.org/abs/2512.05824v1](http://arxiv.org/abs/2512.05824v1)

**作者:** Hafsa Akebli, Adam Shephard, Vincenzo Della Mea, Nasir Rajpoot

**发布时间:** 2025-12-05

**备注:** 4 pages, 2 figures

### GPT解析

### 总结

本研究提出了一种多模态肿瘤学代理(MOA)，用于预测低级别胶质瘤中的IDH1突变，结合了组织学工具和外部生物医学资源推理，实现了高准确率的预测。

### 背景

低级别胶质瘤经常出现IDH1突变，这些突变定义了具有特定预后和治疗意义的临床亚组。

### 目的

开发一种能够准确预测低级别胶质瘤中IDH1突变的多模态肿瘤学代理系统。

### 方法

MOA集成了基于TITAN基础模型的组织学工具，结合对PubMed、Google Search和OncoKB中的结构化临床和基因组输入的推理，并在488名TCGA-LGG队列患者上进行了评估。

### 主要发现

不含组织学工具的MOA F1得分为0.826，优于临床基线(0.798)；与组织学特征融合后，MOA达到最高F1得分0.912，超过了组织学基线(0.894)和融合组织学-临床基线(0.897)。

### 结论

所提出的代理能够捕获通过外部生物医学资源富集的互补突变相关信息，实现了准确的IDH1突变预测。

### 翻译

低级别胶质瘤经常出现IDH1突变，这些突变定义了具有特定预后和治疗意义的临床亚组。本研究引入了一种多模态肿瘤学代理(MOA)，该代理集成了基于TITAN基础模型的用于预测低级别胶质瘤中IDH1突变的组织学工具，并结合对PubMed、Google Search和OncoKB中结构化临床和基因组输入的推理。在488名TCGA-LGG队列患者上对MOA报告进行了定量评估，并与临床和组织学基线进行比较。不含组织学工具的MOA优于临床基线，F1得分为0.826，而临床基线为0.798。当与组织学特征融合后，MOA达到最高性能，F1得分为0.912，超过了组织学基线(0.894)和融合组织学-临床基线(0.897)。这些结果表明，所提出的代理能够捕获通过外部生物医学资源富集的互补突变相关信息，实现了准确的IDH1突变预测。


### 论文摘要

Low-grade gliomas frequently present IDH1 mutations that define clinically distinct subgroups with specific prognostic and therapeutic implications. This work introduces a Multimodal Oncology Agent (MOA) integrating a histology tool based on the TITAN foundation model for IDH1 mutation prediction in low-grade glioma, combined with reasoning over structured clinical and genomic inputs through PubMed, Google Search, and OncoKB. MOA reports were quantitatively evaluated on 488 patients from the TCGA-LGG cohort against clinical and histology baselines. MOA without the histology tool outperformed the clinical baseline, achieving an F1-score of 0.826 compared to 0.798. When fused with histology features, MOA reached the highest performance with an F1-score of 0.912, exceeding both the histology baseline at 0.894 and the fused histology-clinical baseline at 0.897. These results demonstrate that the proposed agent captures complementary mutation-relevant information enriched through external biomedical sources, enabling accurate IDH1 mutation prediction.

---

## 63. Utility Boundary of Dataset Distillation: Scaling and Configuration-Coverage Laws

**论文链接:** [http://arxiv.org/abs/2512.05817v1](http://arxiv.org/abs/2512.05817v1)

**作者:** Zhengquan Luo, Zhiqiang Xu

**发布时间:** 2025-12-05

### GPT解析

### 总结

论文提出了一个名为'配置-动力学-误差分析'的统一理论框架，解决了数据集蒸馏理论基础薄弱的问题，通过缩放定律和覆盖定律解释了数据集蒸馏的性能表现和样本需求，揭示了不同匹配方法的本质联系。

### 背景

数据集蒸馏旨在构建紧凑的合成数据集，使模型能获得与完整数据集训练相当的性能，同时减少存储和计算。尽管实证进展迅速，但现有方法(梯度、分布、轨迹匹配)建立在异构代理目标和优化假设上，缺乏统一理论基础，且不清楚当训练配置变化时蒸馏数据的有效性条件。

### 目的

回答数据集蒸馏的理论问题，提供统一框架解释不同方法工作原理，指导设计紧凑且对配置具有鲁棒性的数据集蒸馏方法。

### 方法

提出'配置-动力学-误差分析'统一理论框架，从共同泛化误差角度重新表述主要数据集蒸馏方法，提供两个主要结果：(1)缩放定律，描述误差随蒸馏样本量增加而减少的规律；(2)覆盖定律，表明所需蒸馏样本量与配置多样性呈线性关系。

### 主要发现

各种匹配方法是可互换的代理，它们减少相同的泛化误差；这解释了为什么不同方法都能实现数据集蒸馏，并提供了关于代理选择如何影响样本效率和鲁棒性的指导。

### 结论

通过不同方法和配置进行的实验确认了推导出的定律，为数据集蒸馏推进了理论基础，使理论驱动的紧凑、配置鲁棒的数据集蒸馏设计成为可能。

### 翻译

数据集蒸馏旨在构建紧凑的合成数据集，使模型能够获得与完整数据集训练相当的性能，同时显著减少存储和计算量。尽管取得了快速的实证进展，但其理论基础仍然有限：现有方法建立在异构的代理目标和优化假设之上，这使得难以分析它们的共同原则或提供一般性保证。为了回答这些问题，我们提出了一个统一理论框架，从共同的泛化误差角度重新表述了主要的数据集蒸馏方法，并提供了缩放定律和覆盖定律两个主要结果。此外，我们的统一分析表明各种匹配方法是可互换的代理，解释了为什么它们都能实现数据集蒸馏，并提供了关于代理选择如何影响样本效率和鲁棒性的指导。实验确认了推导出的定律，为数据集蒸馏推进了理论基础。


### 论文摘要

Dataset distillation (DD) aims to construct compact synthetic datasets that allow models to achieve comparable performance to full-data training while substantially reducing storage and computation. Despite rapid empirical progress, its theoretical foundations remain limited: existing methods (gradient, distribution, trajectory matching) are built on heterogeneous surrogate objectives and optimization assumptions, which makes it difficult to analyze their common principles or provide general guarantees. Moreover, it is still unclear under what conditions distilled data can retain the effectiveness of full datasets when the training configuration, such as optimizer, architecture, or augmentation, changes. To answer these questions, we propose a unified theoretical framework, termed configuration--dynamics--error analysis, which reformulates major DD approaches under a common generalization-error perspective and provides two main results: (i) a scaling law that provides a single-configuration upper bound, characterizing how the error decreases as the distilled sample size increases and explaining the commonly observed performance saturation effect; and (ii) a coverage law showing that the required distilled sample size scales linearly with configuration diversity, with provably matching upper and lower bounds. In addition, our unified analysis reveals that various matching methods are interchangeable surrogates, reducing the same generalization error, clarifying why they can all achieve dataset distillation and providing guidance on how surrogate choices affect sample efficiency and robustness. Experiments across diverse methods and configurations empirically confirm the derived laws, advancing a theoretical foundation for DD and enabling theory-driven design of compact, configuration-robust dataset distillation.

---

## 64. Feasibility study for physics-informed direct numerical simulation describing particle suspension in high-loaded compartments of air-segmented flow

**论文链接:** [http://arxiv.org/abs/2512.05785v1](http://arxiv.org/abs/2512.05785v1)

**作者:** Otto Mierka, Raphael Münster, Henrik Julian Felix Bettin, Kerstin Wohlgemuth, Stefan Turek

**发布时间:** 2025-12-05

**备注:** 26 pages, 10 figures, 4 tables

### GPT解析

### 总结

本研究开发了一种基于有限元-虚边界方法的颗粒分辨直接数值模拟(DNS)框架，用于模拟Archimedes管式结晶器(ATC)中的高密度颗粒悬浮行为。通过引入三个定量指标，成功分类了不同悬浮状态，并验证了DNS能够准确预测实验观察到的流动图区域和细微转变。

### 背景

Archimedes管式结晶器(ATC)使用盘管中的气分段流实现连续结晶的窄停留时间分布，Taylor和Dean涡流驱动颗粒悬浮。然而，单向耦合模型无法在高负载下捕捉关键的流体-颗粒反馈作用。

### 目的

开发能够准确模拟高密度悬浮行为的数值方法，为结晶器设计提供理论基础。

### 方法

提出基于有限元-虚边界方法的颗粒分辨DNS框架，使用硬接触模型模拟颗粒相互作用。对L-丙氨酸悬浮液在不同颗粒大小、固体含量和旋转速度下的模拟通过实验侧视成像验证。引入轴向分布、径向指数和垂直不对称性三个定量指标分类悬浮状态。

### 主要发现

DNS结果成功重现了实验观察到的流动图区域(绿色、黄色、红色/黄色、红色)，并解决了后部负载和垂直对称性丧失等细微转变。DNS能够可靠预测高密度悬浮行为。

### 结论

可行性研究表明DNS可以可靠预测高密度悬浮行为，为结晶器设计提供了力学基础。

### 翻译

阿基米德管式结晶器(ATC)采用盘管中的气分段流实现连续结晶的窄停留时间分布。Taylor涡流和Dean涡流驱动该系统中的颗粒悬浮。然而，单向耦合模型无法捕捉在高负载下变得关键的流体-颗粒反馈作用。我们提出了一种基于有限元-虚边界方法的颗粒分辨直接数值模拟(DNS)框架，使用硬接触模型模拟颗粒相互作用。对不同颗粒大小、固体含量和旋转速度下的L-丙氨酸悬浮液的模拟通过实验侧视成像进行了验证。引入了三个定量指标—轴向分布、径向指数和垂直不对称性—来分类悬浮状态。DNS结果重现了实验观察到的流动图区域(绿色、黄色、红色/黄色、红色)，并解决了后部负载和垂直对称性丧失等细微转变。这项可行性研究表明DNS可以可靠预测高密度悬浮行为，并为结晶器设计提供了力学基础。


### 论文摘要

The Archimedes Tube Crystallizer (ATC) employs air-segmented flow in coiled tubes to achieve narrow residence time distributions for continuous crystallization. Taylor and Dean vortices drive particle suspension in this system. However, one-way coupled models fail to capture the fluid-particle feedback that becomes critical at higher loadings. We present a particle-resolved Direct Numerical Simulation (DNS) framework based on a Finite Element-Fictitious Boundary Method with hard-contact modeling of particle interactions. Simulations of L-alanine suspensions across varying particle sizes, solid contents, and rotational speeds are validated against experimental side-view imaging. Three quantitative metrics-axial distribution, radial index, and vertical asymmetry-are introduced to classify suspension regimes. The DNS results reproduce the experimentally observed flow map zones (green, yellow, red/yellow, red) and resolve subtle transitions such as rear loading and loss of vertical symmetry. This feasibility study demonstrates that DNS can reliably predict dense suspension behavior and provides a mechanistic foundation for crystallizer design.

---

## 65. OWL: Unsupervised 3D Object Detection by Occupancy Guided Warm-up and Large Model Priors Reasoning

**论文链接:** [http://arxiv.org/abs/2512.05698v1](http://arxiv.org/abs/2512.05698v1)

**作者:** Xusheng Guo, Wanfa Zhang, Shijia Zhao, Qiming Xia, Xiaolong Xie, Mingming Wang, Hai Wu, Chenglu Wen

**发布时间:** 2025-12-05

**备注:** The 40th Annual AAAI Conference on Artificial Intelligence

### GPT解析

### 总结

OWL是一种无监督3D目标检测方法，通过占用引导预热和大型模型先验推理解决了现有方法中伪标签不准确的问题。

### 背景

无监督3D目标检测利用启发式算法发现潜在物体，减少自动驾驶标注成本。现有方法生成伪标签并通过自训练迭代优化，但初始伪标签错误会导致优化误导，且有效过滤和优化这些伪标签是关键挑战。

### 目的

提出OWL方法，解决无监督3D目标检测中伪标签不准确的问题，提高检测性能。

### 方法

OWL包含三个核心组件：1)占用引导预热策略(OGW)初始化具有空间感知能力的骨干网络权重；2)基于实例的推理模块(ICR)利用大型模型先验知识评估伪标签质量；3)自适应权重的自训练策略(WAS)动态重新加权伪标签。

### 主要发现

在Waymo Open Dataset和KITTI上的实验表明，OWL比最先进的无监督方法性能超过15.0% mAP。

### 结论

OWL方法在无监督3D目标检测领域表现优异，其有效性得到实验验证。

### 翻译

无监督3D目标检测利用启发式算法来发现潜在物体，为减少自动驾驶标注成本提供了一条有前景的途径。现有方法主要生成伪标签并通过自训练迭代进行优化。然而，这些伪标签在训练初期往往是错误的，导致优化过程被误导。此外，有效过滤和优化这些伪标签仍然是一个关键挑战。在本文中，我们提出OWL用于无监督3D目标检测，通过占用引导预热和大型模型先验推理。OWL首先采用占用引导预热(OGW)策略来初始化具有空间感知能力的骨干网络权重，减轻不正确伪标签对网络收敛的干扰。此外，OWL引入了一个基于实例的推理(ICR)模块，利用大型模型的先验知识评估伪标签质量，实现精确的过滤和优化。最后，我们设计了一个自适应权重的自训练(WAS)策略，通过动态重新加权伪标签来提升性能。在Waymo Open Dataset(WOD)和KITTI上的大量实验表明，OWL比最先进的无监督方法性能超过15.0% mAP，揭示了我们方法的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决无监督3D目标检测中的两个关键挑战：1)不稳定的网络初始化问题，训练初期网络对不准确的伪标签高度敏感；2)伪标签的过滤和精炼问题，错误标签在自训练过程中会累积和放大。这个问题在现实中非常重要，因为3D目标检测对自动驾驶至关重要，而现有的监督方法严重依赖昂贵且耗时的人工标注，无监督方法可以大幅降低这些成本。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有无监督3D目标检测方法的局限性设计了OWL框架。对于网络初始化问题，借鉴了占用预测任务(如Occupancy-MAE)但改进为专门针对无监督场景的OGW策略；对于伪标签精炼问题，创新性地引入了大模型先验推理能力，不同于传统基于阈值的过滤方法；还设计了WAS策略改进自训练过程。作者借鉴了多种现有工作，包括DBSCAN聚类、自监督学习、大模型推理等，但进行了针对性改进以适应无监督3D目标检测的特殊需求。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过三个组件协同解决无监督3D目标检测的挑战：1)利用占用预测进行网络预热，使网络具备初步空间感知能力；2)利用大模型先验知识评估和过滤伪标签；3)通过动态权重分配优化自训练。整体流程：首先通过运动伪影去除和动态半径聚类生成初始伪标签；然后使用OGW预热网络；接着利用ICR模块提取实例提示并通过大模型推理精炼伪标签；最后通过WAS策略进行自适应自训练，迭代优化检测器。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)占用引导预热(OGW)策略，利用占用预测任务初始化网络，使其具备空间感知能力；2)实例提示推理(ICR)模块，利用大模型先验知识和多种实例提示评估伪标签质量；3)权重自适应自训练(WAS)策略，动态重新加权伪标签。相比之前的工作，OWL的不同之处在于：专门针对无监督3D目标检测的两个核心挑战设计解决方案；创新性地将大模型推理能力引入3D目标检测的伪标签精炼；采用多阶段策略，每个阶段针对特定问题；在多个数据集上显著优于之前方法，提升超过15% mAP。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'OWL通过占用引导预热网络初始化、利用大模型先验知识精炼伪标签以及动态权重自适应自训练，显著提升了无监督3D目标检测的性能，减少了自动驾驶领域对昂贵人工标注的依赖。'}


### 论文摘要

Unsupervised 3D object detection leverages heuristic algorithms to discover potential objects, offering a promising route to reduce annotation costs in autonomous driving. Existing approaches mainly generate pseudo labels and refine them through self-training iterations. However, these pseudo-labels are often incorrect at the beginning of training, resulting in misleading the optimization process. Moreover, effectively filtering and refining them remains a critical challenge. In this paper, we propose OWL for unsupervised 3D object detection by occupancy guided warm-up and large-model priors reasoning. OWL first employs an Occupancy Guided Warm-up (OGW) strategy to initialize the backbone weight with spatial perception capabilities, mitigating the interference of incorrect pseudo-labels on network convergence. Furthermore, OWL introduces an Instance-Cued Reasoning (ICR) module that leverages the prior knowledge of large models to assess pseudo-label quality, enabling precise filtering and refinement. Finally, we design a Weight-adapted Self-training (WAS) strategy to dynamically re-weight pseudo-labels, improving the performance through self-training. Extensive experiments on Waymo Open Dataset (WOD) and KITTI demonstrate that OWL outperforms state-of-the-art unsupervised methods by over 15.0% mAP, revealing the effectiveness of our method.

---

## 66. HiMoE-VLA: Hierarchical Mixture-of-Experts for Generalist Vision-Language-Action Policies

**论文链接:** [http://arxiv.org/abs/2512.05693v1](http://arxiv.org/abs/2512.05693v1)

**作者:** Zhiying Du, Bei Liu, Yaobo Liang, Yichao Shen, Haidong Cao, Xiangyu Zheng, Zhiyuan Feng, Zuxuan Wu, Jiaolong Yang, Yu-Gang Jiang

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种名为HiMoE-VLA的新型视觉-语言-动作(VLA)框架，用于有效处理具有异质性的多样化机器人数据。通过引入分层专家混合(HiMoE)架构，该方法能够自适应处理多种异质性源，并将其逐步抽象为共享知识表示，在多种机器人和动作空间中实现了更高的准确性和鲁棒泛化能力。

### 背景

具身智能基础模型的发展依赖于大规模、高质量的机器人演示数据，但机器人演示数据在 embodiment、动作空间以及传感器配置和动作控制频率等方面存在显著异质性，这使得现有方法难以有效整合这些多样化因素。

### 目的

开发一种能够有效处理具有异质性的多样化机器人数据的框架，以提高具身智能基础模型的泛化能力和在新环境中的性能表现。

### 方法

提出HiMoE-VLA框架，其中包含一个针对动作模块的分层专家混合(HiMoE)架构，该架构能够自适应处理各层级中的多种异质性源，并逐步将它们抽象为共享的知识表示。

### 主要发现

在仿真基准和真实机器人平台上的广泛实验表明，HiMoE-VLA比现有的VLA基线方法表现更好，在多样化的机器人和动作空间中实现了更高的准确性和鲁棒泛化能力。

### 结论

HiMoE-VLA框架通过有效处理机器人数据中的异质性，显著提升了具身智能基础模型的性能和泛化能力，为具身智能领域的发展提供了新的解决方案。

### 翻译

本文提出了一种名为HiMoE-VLA的新型视觉-语言-动作(VLA)框架，用于有效处理具有异质性的多样化机器人数据。通过引入分层专家混合(HiMoE)架构，该方法能够自适应处理多种异质性源，并将其逐步抽象为共享知识表示，在多种机器人和动作空间中实现了更高的准确性和鲁棒泛化能力。代码和模型已在GitHub上公开。


### 论文摘要

The development of foundation models for embodied intelligence critically depends on access to large-scale, high-quality robot demonstration data. Recent approaches have sought to address this challenge by training on large collections of heterogeneous robotic datasets. However, unlike vision or language data, robotic demonstrations exhibit substantial heterogeneity across embodiments and action spaces as well as other prominent variations such as senor configurations and action control frequencies. The lack of explicit designs for handling such heterogeneity causes existing methods to struggle with integrating diverse factors, thereby limiting their generalization and leading to degraded performance when transferred to new settings. In this paper, we present HiMoE-VLA, a novel vision-language-action (VLA) framework tailored to effectively handle diverse robotic data with heterogeneity. Specifically, we introduce a Hierarchical Mixture-of-Experts (HiMoE) architecture for the action module which adaptively handles multiple sources of heterogeneity across layers and gradually abstracts them into shared knowledge representations. Through extensive experimentation with simulation benchmarks and real-world robotic platforms, HiMoE-VLA demonstrates a consistent performance boost over existing VLA baselines, achieving higher accuracy and robust generalization across diverse robots and action spaces. The code and models are publicly available at https://github.com/ZhiyingDu/HiMoE-VLA.

---

## 67. Interleaved Latent Visual Reasoning with Selective Perceptual Modeling

**论文链接:** [http://arxiv.org/abs/2512.05665v1](http://arxiv.org/abs/2512.05665v1)

**作者:** Shuai Dong, Siyuan Wang, Xingyu Liu, Zhongyu Wei

**发布时间:** 2025-12-05

**备注:** 11 pages, 6 figures. Code available at https://github.com/XD111ds/ILVR

### GPT解析

### 总结

本文提出了一种名为交错潜在视觉推理(ILVR)的框架，解决了多模态大语言模型在视觉推理中的计算效率和精确感知之间的权衡问题。

### 背景

交错推理范式通过视觉反馈增强多模态大语言模型，但反复编码像素密集图像的计算成本过高；潜在视觉推理虽可解决此瓶颈，但存在过度压缩特征牺牲精确感知或静态结构无法建模动态问题的权衡。

### 目的

开发一种统一动态状态演化和精确感知建模的框架，以克服现有方法的局限性。

### 方法

提出交错潜在视觉推理(ILVR)框架，将文本生成与潜在视觉表示交错，作为后续推理的具体、演化线索；采用自监督策略，通过动量教师模型有选择地将辅助图像的相关特征蒸馏到稀疏监督目标中，引导模型自主生成上下文感知的视觉信号。

### 主要发现

在多模态推理基准上的广泛实验表明，ILVR显著优于现有方法，有效弥合了细粒度感知和顺序多模态推理之间的差距。

### 结论

ILVR成功结合了精确感知建模和动态推理能力，为多模态大语言模型的高效视觉推理提供了新思路。

### 翻译

交错推理范式通过视觉反馈增强多模态大语言模型(MLLMs)，但由于反复编码像素密集图像的 prohibitive 计算成本而受到阻碍。一种有前景的替代方法——潜在视觉推理，绕过了这一瓶颈，但目前却迫使一个关键的权衡：方法要么通过过度压缩特征而牺牲精确感知建模，要么由于静态、非交错结构而无法建模动态问题。我们引入了交错潜在视觉推理(ILVR)，这是一个统一动态状态演化和精确感知建模的框架。ILVR将文本生成与潜在视觉表示交错，这些表示作为后续推理的具体、演化的线索。为此，我们采用了一种自监督策略，其中动量教师模型有选择地将辅助图像的相关特征蒸馏到稀疏监督目标中。这种自适应选择机制引导模型自主生成上下文感知的视觉信号。在多模态推理基准上的广泛实验表明，ILVR显著优于现有方法，有效弥合了细粒度感知和顺序多模态推理之间的差距。


### 论文摘要

Interleaved reasoning paradigms enhance Multimodal Large Language Models (MLLMs) with visual feedback but are hindered by the prohibitive computational cost of repeatedly re-encoding pixel-dense images. A promising alternative, latent visual reasoning, circumvents this bottleneck yet currently forces a critical trade-off: methods either sacrifice precise perceptual modeling by over-compressing features or fail to model dynamic problems due to static, non-interleaved structures. We introduce Interleaved Latent Visual Reasoning (ILVR), a framework that unifies dynamic state evolution with precise perceptual modeling. ILVR interleaves textual generation with latent visual representations that act as specific, evolving cues for subsequent reasoning. To enable this, we employ a self-supervision strategy where a Momentum Teacher Model selectively distills relevant features from helper images into sparse supervision targets. This adaptive selection mechanism guides the model to autonomously generate context-aware visual signals. Extensive experiments on multimodal reasoning benchmarks demonstrate that ILVR significantly outperforms existing approaches, effectively bridging the gap between fine-grained perception and sequential multimodal reasoning.

---

## 68. Standard and stressed value at risk forecasting using dynamic Bayesian networks

**论文链接:** [http://arxiv.org/abs/2512.05661v1](http://arxiv.org/abs/2512.05661v1)

**作者:** Eden Gross, Ryan Kruger, Francois Toerien

**发布时间:** 2025-12-05

**备注:** 30 pages, 4 tables (excluding appendix, in which there is one table)

### GPT解析

### 总结

本研究提出了一种动态贝叶斯网络框架用于预测风险价值和压力风险价值，并与传统模型进行了性能比较。

### 背景

风险价值（VaR）和压力风险价值（SVaR）是金融风险管理中的重要指标，需要准确的预测方法。

### 目的

比较动态贝叶斯网络（DBN）框架与传统模型在预测VaR和SVaR方面的性能。

### 方法

使用1991年至2020年的标准普尔500指数日收益率数据，通过滚动期和历史收益率为传统模型生成10天99%的VaR和SVaR预测，同时使用三个DBN模型结合历史和预测收益率，并使用标准回测和预测误差指标评估模型性能。

### 主要发现

自回归模型提供了最准确的VaR预测；DBN的表现与历史模拟模型相当，尽管纳入了前瞻性收益预测；对于SVaR，所有模型都产生了高度保守的预测，突破较少且准确性差异有限。

### 结论

尽管DBN没有超越传统模型，但它们展示了作为前瞻性方法的可行性，为将因果推理整合到金融风险预测的未来研究奠定了基础。

### 翻译

本研究引入了一种用于预测风险价值（VaR）和压力风险价值（SVaR）的动态贝叶斯网络（DBN）框架，并将其性能与几种常用应用模型进行了比较。使用1991年至2020年的标准普尔500指数日收益率，我们通过滚动期和历史收益率为传统模型生成10天99%的VaR和SVaR预测，而三个DBN则同时使用历史和预测收益率。我们使用标准回测和预测误差指标评估模型的预测准确性。结果表明，自回归模型提供了最准确的VaR预测，而DBN的表现与历史模拟模型相当，尽管它们纳入了前瞻性的收益预测。对于SVaR，所有模型都产生了高度保守的预测，突破较少且准确性差异有限。尽管DBN没有超越传统模型，但它们展示了作为前瞻性方法的可行性，为将因果推理整合到金融风险预测的未来研究奠定了基础。


### 论文摘要

This study introduces a dynamic Bayesian network (DBN) framework for forecasting value at risk (VaR) and stressed VaR (SVaR) and compares its performance to several commonly applied models. Using daily S&P 500 index returns from 1991 to 2020, we produce 10-day 99% VaR and SVaR forecasts using a rolling period and historical returns for the traditional models, while three DBNs use both historical and forecasted returns. We evaluate the models' forecasting accuracy using standard backtests and forecasting error measures. Results show that autoregressive models deliver the most accurate VaR forecasts, while the DBNs achieve comparable performance to the historical simulation model, despite incorporating forward-looking return forecasts. For SVaR, all models produce highly conservative forecasts, with minimal breaches and limited differentiation in accuracy. While DBNs do not outperform traditional models, they demonstrate feasibility as a forward-looking approach to provide a foundation for future research on integrating causal inference into financial risk forecasting.

---

## 69. A Greek Government Decisions Dataset for Public-Sector Analysis and Insight

**论文链接:** [http://arxiv.org/abs/2512.05647v1](http://arxiv.org/abs/2512.05647v1)

**作者:** Giorgos Antoniou, Giorgos Filandrianos, Aggelos Vlachos, Giorgos Stamou, Lampros Kollimenos, Konstantinos Skianis, Michalis Vazirgiannis

**发布时间:** 2025-12-05

### GPT解析

### 总结

本研究创建了一个开放、机器可读的希腊政府决策语料库，包含100万份决策，提供了高质量的原始文本和完全可重现的提取流程。研究还进行了定性分析，设计了检索增强生成任务，并评估了基线系统，展示了该资源在支持高级信息访问、透明度和语言模型开发方面的潜力。

### 背景

研究基于国家透明度平台Diavgeia，该平台包含大量希腊政府决策文档。这些文档通常以PDF格式存储，难以进行系统性的文本分析和处理，因此需要创建一个结构化的、机器可读的语料库。

### 目的

创建一个高质量的资源以支持对政府决策的研究；探索政府文件中的模板模式；设计并评估检索增强生成(RAG)任务；展示大规模公共部门语料库在支持高级信息访问和透明度方面的潜力；为语言模型提供高价值的预训练或微调材料。

### 方法

从PDF中提取高质量原始文本并以Markdown格式发布；提供完全可重现的提取流程；进行定性分析以探索模板模式；制定代表性问题集并创建高质量答案；评估基线RAG系统检索和推理公共决策的能力；讨论局限性和未来方向。

### 主要发现

评估展示了大规模公共部门语料库支持高级信息访问和透明度的潜力；RAG管道可以模拟基于聊天的助手，能够交互式地回答关于公共决策的问题；该语料库由于规模、质量和领域覆盖度，可作为法律和政府领域专业模型的高价值预训练或微调材料。

### 结论

该语料库可作为法律和政府领域专业模型的基础；作为领域适应、知识生成和可解释AI新方法的基础；作者通过使数据和代码可访问，促进了进一步的研究和应用开发。

### 翻译

我们介绍了一个开放、机器可读的希腊政府决策语料库，源自国家透明度平台Diavgeia。该资源包含100万份决策，具有从PDF中提取的高质量原始文本。它以Markdown格式发布原始提取文本，并提供完全可重现的提取流程。除了核心数据集外，我们进行了定性分析以探索模板模式，并通过制定一组代表性问题、创建高质量答案以及评估基线RAG系统检索和推理公共决策的能力，设计了检索增强生成(RAG)任务。这一评估展示了大规模公共部门语料库通过结构化检索和推理政府文件来支持高级信息访问和透明度的潜力，并突出了此类RAG管道如何能够模拟聊天式助手，能够交互式地回答关于公共决策的问题。由于其规模、质量和领域覆盖度，该语料库也可作为新语言模型和高价值预训练或微调材料，包括法律和政府领域的专业模型，以及作为领域适应、知识生成和可解释AI新方法的基础。最后，我们讨论了局限性，概述了未来方向，并使数据和代码均可访问。


### 论文摘要

We introduce an open, machine-readable corpus of Greek government decisions sourced from the national transparency platform Diavgeia. The resource comprises 1 million decisions, featuring and high-quality raw text extracted from PDFs. It is released with raw extracted text in Markdown format, alongside a fully reproducible extraction pipeline. Beyond the core dataset, we conduct qualitative analyses to explore boilerplate patterns and design a retrieval-augmented generation (RAG) task by formulating a set of representative questions, creating high-quality answers, and evaluating a baseline RAG system on its ability to retrieve and reason over public decisions. This evaluation demonstrates the potential of large-scale public-sector corpora to support advanced information access and transparency through structured retrieval and reasoning over governmental documents, and highlights how such a RAG pipeline could simulate a chat-based assistant capable of interactively answering questions about public decisions. Due to its scale, quality, and domain coverage, the corpus can also serve as high-value pre-training or fine-tuning material for new Language Models (LMs) and Large Language Models (LLMs) respectively, including specialized models for legal and governmental domains, and as a foundation for novel approaches in domain adaptation, knowledge-grounded generation, and explainable AI. Finally, we discuss limitations, outline future directions, and make both the data and the code accessible.

---

## 70. On the Impact of the Communication Model on Realisability

**论文链接:** [http://arxiv.org/abs/2512.05609v1](http://arxiv.org/abs/2512.05609v1)

**作者:** Cinzia Di Giusto, Etienne Lozes, Pascal Urso

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文开发了一个统一框架，用于推理跨多种通信模型的可实现性和子类型关系，研究表明通信模型不影响子类型关系但影响可实现性。

### 背景

多方会话类型(MPST)为分布式系统中的通信协议规范和验证提供了类型理论基础，依赖于全局类型(指定全局行为)和局部类型(全局行为在本地参与者上的投影)。

### 目的

解决在非对等对等(P2P)通信模型(如基于包的、因果有序的或同步通信)中MPST可实现性理解不足的问题。

### 方法

开发统一框架来推理跨多种通信模型的可实现性和子类型关系，引入多种决策程序进行子类型检查和可实现性检查。

### 主要发现

通信模型不影响子类型关系的概念，但影响可实现性的概念；引入的决策程序复杂度从NLOGSPACE到EXPSPACE不等，取决于全局类型的可补性和补集大小。

### 结论

通过统一框架，作者扩展了对不同通信模型下MPST的理解，明确了通信模型对子类型关系和可实现性的不同影响。

### 翻译

多方会话类型(MPST)为分布式系统中的通信协议规范和验证提供了类型理论基础。MPST依赖于全局类型的概念，它指定了全局行为，以及局部类型，它是全局行为在每个本地参与者上的投影。MPST中的核心概念是可实现性-即从全局规范导出的局部实现在给定通信模型下是否正确实现了预期协议。虽然对等对等语义下的可实现性已被广泛研究，但在其他通信模型(如基于包的、因果有序的或同步通信)中仍缺乏理解。在本文中，我们开发了一个统一框架，用于推理跨多种通信模型的可实现性和子类型关系。我们表明通信模型不影响子类型关系的概念，但影响可实现性的概念。我们引入了几种子类型检查和可实现性检查的决策过程，复杂度从NLOGSPACE到EXPSPACE不等，这取决于对全局类型的假设，特别是它们的可补性和给定补集的大小。


### 论文摘要

Multiparty Session Types (MPST) provide a type-theoretic foundation for specifying and verifying communication protocols in distributed systems. MPST rely on the notion of global type which specifies the global behaviour and local types, which are the projections of the global behaviour onto each local participant. A central notion in MPST is realisability - whether local implementations derived from a global specification correctly realise the intended protocol under a given communication model. While realisability has been extensively studied under peer-to-peer semantics, it remains poorly understood in alternative communication models such as bag-based, causally ordered, or synchronous communications. In this paper, we develop a unified framework for reasoning about realisability and subtyping across a spectrum of communication models. We show that the communication model does not impact the notion of subtyping, but that it impacts the notion of realisability. We introduce several decision procedures for subtyping checking and realisability checking with complexities ranging from NLOGSPACE to EXPSPACE depending on the assumptions made on the global types, in particular depending on their complementability and the size of a given complement.

---

## 71. CureAgent: A Training-Free Executor-Analyst Framework for Clinical Reasoning

**论文链接:** [http://arxiv.org/abs/2512.05576v1](http://arxiv.org/abs/2512.05576v1)

**作者:** Ting-Ting Xie, Yixin Zhang

**发布时间:** 2025-12-05

**备注:** 2nd Place Solution to the CURE-Bench Competition @ NeurIPS 2025. Code available at https://github.com/June01/CureAgent

### GPT解析

### 总结

本文提出了一种Executor-Analyst Framework来解决临床代理中的上下文利用失败问题，通过模块化架构将工具执行与临床推理解耦，并采用分层集成策略提高性能。研究发现存在上下文-性能悖论和维度诅咒问题，该方法无需昂贵的端到端微调即可实现最先进的性能。

### 背景

当前基于小型语言模型(LLMs)的临床代理(如TxAgent)存在上下文利用失败问题，模型能够检索到生物医学证据，但无法将这些证据作为诊断的基础。

### 目的

提出一种新的框架来解决临床代理中的上下文利用失败问题，提高临床推理的准确性和可靠性。

### 方法

提出Executor-Analyst Framework(执行器-分析师框架)，这是一种模块化架构，将工具执行的语法精确性与临床推理的语义稳健性解耦。通过协调专门的TxAgent(执行器)与长上下文基础模型(分析师)，减轻了整体模型中观察到的推理缺陷。此外，采用分层集成策略，通过保留证据多样性，显著优于全局池化方法，有效解决了信息瓶颈问题。

### 主要发现

1) 存在上下文-性能悖论，将推理上下文扩展到12k以上会引入噪声并降低准确性；2) 观察到维度诅咒在行动空间中的表现，扩展工具集需要分层检索策略。

### 结论

该方法强调了无需训练的架构工程的潜力，在CURE-Bench上实现了最先进的性能，无需昂贵的端到端微调。这为下一代可信AI驱动的治疗提供了可扩展、灵活的基础。

### 翻译

当前基于小型语言模型(LLMs)的临床代理，如TxAgent，遭受上下文利用失败的困扰，模型能够成功检索生物医学证据，但无法将这些信息作为诊断基础。在本工作中，我们提出了Executor-Analyst Framework，一种模块化架构，将工具执行的语法精确性与临床推理的语义稳健性解耦。通过协调专门的TxAgent(执行器)与长上下文基础模型(分析师)，我们减轻了整体模型中观察到的推理缺陷。除了简单的模块化外，我们证明分层集成策略通过保留证据多样性显著优于全局池化，有效解决了信息瓶颈问题。此外，我们的压力测试揭示了关键的扩展见解：(1) 上下文-性能悖论，将推理上下文扩展到12k以上会引入噪声并降低准确性；(2) 行动空间中的维度诅咒，扩展工具集需要分层检索策略。关键的是，我们的方法强调了无需训练的架构工程的潜力，在CURE-Bench上实现了最先进的性能，无需昂贵的端到端微调。这为下一代可信AI驱动的治疗提供了可扩展、灵活的基础。代码已在https://github.com/June01/CureAgent上发布。


### 论文摘要

Current clinical agent built on small LLMs, such as TxAgent suffer from a \textit{Context Utilization Failure}, where models successfully retrieve biomedical evidence due to supervised finetuning but fail to ground their diagnosis in that information. In this work, we propose the Executor-Analyst Framework, a modular architecture that decouples the syntactic precision of tool execution from the semantic robustness of clinical reasoning. By orchestrating specialized TxAgents (Executors) with long-context foundation models (Analysts), we mitigate the reasoning deficits observed in monolithic models. Beyond simple modularity, we demonstrate that a Stratified Ensemble strategy significantly outperforms global pooling by preserving evidentiary diversity, effectively addressing the information bottleneck. Furthermore, our stress tests reveal critical scaling insights: (1) a \textit{Context-Performance Paradox}, where extending reasoning contexts beyond 12k tokens introduces noise that degrades accuracy; and (2) the \textit{Curse of Dimensionality} in action spaces, where expanding toolsets necessitates hierarchical retrieval strategies. Crucially, our approach underscores the potential of training-free architectural engineering, achieving state-of-the-art performance on CURE-Bench without the need for expensive end-to-end finetuning. This provides a scalable, agile foundation for the next generation of trustworthy AI-driven therapeutics. Code has been released on https://github.com/June01/CureAgent.

---

## 72. ProPhy: Progressive Physical Alignment for Dynamic World Simulation

**论文链接:** [http://arxiv.org/abs/2512.05564v1](http://arxiv.org/abs/2512.05564v1)

**作者:** Zijun Wang, Panwen Hu, Jing Wang, Terry Jingchen Zhang, Yuhao Cheng, Long Chen, Yiqiang Yan, Zutao Jiang, Hanhui Li, Xiaodan Liang

**发布时间:** 2025-12-05

### GPT解析

### 总结

ProPhy是一种渐进式物理对齐框架，通过两阶段的物理专家混合机制实现物理感知的视频生成，解决了现有模型在处理大规模或复杂动力学时物理一致性的问题。

### 背景

视频生成技术在构建世界模拟器方面显示出巨大潜力，但当前模型在处理大规模或复杂动力学时仍难以产生物理一致的结果。这是因为现有方法对物理提示的响应是各向同性的，并且忽视了生成内容与局部物理线索之间的细粒度对齐。

### 目的

为了解决这些挑战，作者提出了ProPhy，一个渐进式物理对齐框架，可以实现明确的物理感知条件和各向异性生成。

### 方法

ProPhy采用两阶段的物理专家混合（Mixture-of-Physics-Experts, MoPE）机制进行判别式物理先验提取。其中，语义专家从文本描述中推断语义级别的物理原理，而细化专家捕获令牌级别的物理动力学。此外，作者还引入了一种物理对齐策略，将视觉语言模型的物理推理能力转移到细化专家中，从而更准确地表示动态物理现象。

### 主要发现

在物理感知视频生成基准上的大量实验表明，ProPhy比现有最先进的方法产生更真实、动态和物理一致的结果。

### 结论

ProPhy通过明确的物理感知条件和各向异性生成，解决了当前视频生成模型在物理一致性方面的局限性。

### 翻译

最近的视频生成进展在构建世界模拟器方面显示出巨大潜力。然而，当前模型在产生物理一致的结果时仍然存在困难，特别是在处理大规模或复杂动力学时。这种限制主要是因为现有方法对物理提示的响应是各向同性的，并且忽视了生成内容与局部物理线索之间的细粒度对齐。为了解决这些挑战，我们提出了ProPhy，一个渐进式物理对齐框架，能够实现明确的物理感知条件和各向异性生成。ProPhy采用两阶段的物理专家混合（MoPE）机制进行判别式物理先验提取，其中语义专家从文本描述中推断语义级别的物理原理，而细化专家捕获令牌级别的物理动力学。这种机制使模型能够学习更好地反映底层物理定律的细粒度、物理感知的视频表示。此外，我们引入了一种物理对齐策略，将视觉语言模型的物理推理能力转移到细化专家中，促进对动态物理现象更准确的表示。在物理感知视频生成基准上的大量实验表明，ProPhy比现有的最先进方法产生更真实、动态和物理一致的结果。


### 论文摘要

Recent advances in video generation have shown remarkable potential for constructing world simulators. However, current models still struggle to produce physically consistent results, particularly when handling large-scale or complex dynamics. This limitation arises primarily because existing approaches respond isotropically to physical prompts and neglect the fine-grained alignment between generated content and localized physical cues. To address these challenges, we propose ProPhy, a Progressive Physical Alignment Framework that enables explicit physics-aware conditioning and anisotropic generation. ProPhy employs a two-stage Mixture-of-Physics-Experts (MoPE) mechanism for discriminative physical prior extraction, where Semantic Experts infer semantic-level physical principles from textual descriptions, and Refinement Experts capture token-level physical dynamics. This mechanism allows the model to learn fine-grained, physics-aware video representations that better reflect underlying physical laws. Furthermore, we introduce a physical alignment strategy that transfers the physical reasoning capabilities of vision-language models (VLMs) into the Refinement Experts, facilitating a more accurate representation of dynamic physical phenomena. Extensive experiments on physics-aware video generation benchmarks demonstrate that ProPhy produces more realistic, dynamic, and physically coherent results than existing state-of-the-art methods.

---

## 73. A Unified AI System For Data Quality Control and DataOps Management in Regulated Environments

**论文链接:** [http://arxiv.org/abs/2512.05559v1](http://arxiv.org/abs/2512.05559v1)

**作者:** Devender Saini, Bhavika Jain, Nitish Ujjwal, Philip Sommer, Dan Romuald Mbanga, Dhagash Mehta

**发布时间:** 2025-12-05

**备注:** 10 pages, 9 figures, 5 tables

### GPT解析

### 总结

本研究提出了一种统一的AI驱动的数据质量控制框架，将质量控制整合为系统核心组件而非孤立步骤，在金融监管环境中实现了更好的异常检测、减少人工修复工作并提高可审计性。

### 背景

在金融等受监管领域，数据管道的完整性和治理至关重要，然而现有系统将数据质量控制(QC)视为一个孤立的预处理步骤，而不是一个一流的系统组件。

### 目的

开发一个统一的AI驱动的数据质量控制和数据运营管理框架，将基于规则、统计和AI的QC方法嵌入到连续的、受管理的层中，贯穿数据摄取、模型管道和下游应用。

### 方法

构建集成开源工具与自定义模块的架构，用于数据剖析、审计日志、违规处理、基于配置的策略和动态修复；在生产级金融环境中部署，处理跨多个资产类别的流式和表格数据，配备可配置阈值、云原生存储接口和自动警报。

### 主要发现

实现了异常检测召回率的实证提升、减少了手动修复工作、提高了高吞吐量数据工作流中的可审计性和可追溯性。

### 结论

通过将QC视为系统关注事项而非事后考虑，该框架为受监管环境中的可信、可扩展和合规的AI管道奠定了基础。

### 翻译

在金融等受监管领域，数据管道的完整性和治理至关重要-然而现有系统将数据质量控制(QC)视为一个孤立的预处理步骤，而不是一个一流的系统组件。我们提出了一个统一的AI驱动的数据质量控制(Data QC)和数据运营管理(DataOps Management)框架，将基于规则、统计和AI的QC方法嵌入到一个连续的、受管理的层中，该层贯穿数据摄取、模型管道和下游应用。我们的架构将开源工具与自定义模块集成，用于数据剖析、审计日志、违规处理、基于配置的策略和动态修复。我们在生产级金融环境中展示了部署情况：处理跨多个资产类别和交易流的流式和表格数据，具有可配置的阈值、云原生存储接口和自动警报。我们展示了异常检测召回率的实证提升、手动修复工作的减少以及在高吞吐量数据工作流中可审计性和可追溯性的改善。通过将QC视为系统关注事项而非事后考虑，我们的框架为受监管环境中的可信、可扩展和合规的AI管道奠定了基础。


### 论文摘要

In regulated domains such as finance, the integrity and governance of data pipelines are critical - yet existing systems treat data quality control (QC) as an isolated preprocessing step rather than a first-class system component. We present a unified AI-driven Data QC and DataOps Management framework that embeds rule-based, statistical, and AI-based QC methods into a continuous, governed layer spanning ingestion, model pipelines, and downstream applications. Our architecture integrates open-source tools with custom modules for profiling, audit logging, breach handling, configuration-driven policies, and dynamic remediation. We demonstrate deployment in a production-grade financial setup: handling streaming and tabular data across multiple asset classes and transaction streams, with configurable thresholds, cloud-native storage interfaces, and automated alerts. We show empirical gains in anomaly detection recall, reduction of manual remediation effort, and improved auditability and traceability in high-throughput data workflows. By treating QC as a system concern rather than an afterthought, our framework provides a foundation for trustworthy, scalable, and compliant AI pipelines in regulated environments.

---

## 74. Conscious Gaze: Adaptive Attention Mechanisms for Hallucination Mitigation in Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2512.05546v1](http://arxiv.org/abs/2512.05546v1)

**作者:** Weijue Bu, Guan Yuan, Guixian Zhang

**发布时间:** 2025-12-05

**备注:** 6 pages, 6 figures

### GPT解析

### 总结

大型视觉-语言模型(VLMs)常表现出文本惯性现象，注意力从视觉证据转向语言先验，导致对象幻觉。现有解码策略仅能在输出层面干预，无法纠正内部推理漂移，而最近的内部控制方法缺乏理论基础。作者提出了Conscious Gaze (CG-VLM)框架，将博弈论可解释性转化为解码控制，通过认知需求传感器和聚焦共识归纳模块实现精确干预，在各种模型和数据集上取得最先进结果。

### 背景

大型视觉-语言模型(VLMs)存在文本惯性问题，即注意力从视觉证据转向语言先验，导致对象幻觉。现有解码策略仅能在输出logits层面干预，无法纠正内部推理漂移，而最近的内部控制方法基于启发式头抑制或全局转向向量，缺乏理论依据。

### 目的

引入Conscious Gaze (CG-VLM)框架，这是一个无需训练、仅在推理时使用的框架，将博弈论可解释性转化为可操作的解码控制，以解决VLMs中的文本惯性问题。

### 方法

CG-VLM包含两个主要组件：1) 基于Harsanyi交互的认知需求传感器，估算即时的视觉-文本协同作用，识别需要视觉锚定的时刻；2) 聚焦共识归纳模块，根据传感器信号有选择地将中间层注意力重新定向到视觉token，防止其坍缩为文本先验。

### 主要发现

CG-VLM在POPE和CHAIR数据集上取得了最先进的结果，适用于InstructBLIP、LLaVA、Qwen-VL和mPLUG等多种模型，同时保留了通用能力。

### 结论

token级别的感知能够实现精确、上下文感知的干预，而不会损害模型的基础知识，证明了理论指导的内部控制方法的有效性。

### 翻译

大型视觉-语言模型(VLMs)通常表现出文本惯性，即注意力从视觉证据转向语言先验，导致对象幻觉。现有的解码策略仅在输出logits层面进行干预，因此无法纠正内部推理漂移，而最近基于启发式头抑制或全局转向向量的内部控制方法缺乏理论依据。我们引入了Conscious Gaze (CG-VLM)，这是一个无需训练、仅在推理时使用的框架，将博弈论可解释性转化为可操作的解码控制。基于Harsanyi交互构建的认知需求传感器估算即时的视觉-文本协同作用，并识别需要视觉锚定的时刻。基于此信号，聚焦共识归纳模块有选择地将中间层注意力重新定向到视觉token，防止其坍缩为文本先验。CG-VLM在POPE和CHAIR数据集上取得了最先进的结果，适用于InstructBLIP、LLaVA、Qwen-VL和mPLUG，同时保留了通用能力，证明了token级别的感知能够实现精确、上下文感知的干预，而不会损害基础知识。


### 论文摘要

Large Vision-Language Models (VLMs) often exhibit text inertia, where attention drifts from visual evidence toward linguistic priors, resulting in object hallucinations. Existing decoding strategies intervene only at the output logits and thus cannot correct internal reasoning drift, while recent internal-control methods based on heuristic head suppression or global steering vectors lack principled grounding. We introduce Conscious Gaze (CG-VLM), a training-free, inference-time framework that converts game-theoretic interpretability into actionable decoding control. A Cognitive Demand Sensor built on Harsanyi interactions estimates instantaneous vision-text synergy and identifies moments when visual grounding is necessary. Conditioned on this signal, a Focused Consensus Induction module selectively reorients mid-layer attention toward visual tokens before collapse into text priors. CG-VLM achieves state-of-the-art results on POPE and CHAIR across InstructBLIP, LLaVA, Qwen-VL, and mPLUG, while preserving general capabilities, demonstrating that token-level sensing enables precise, context-aware intervention without compromising foundational knowledge.

---

## 75. On the Theoretical Foundation of Sparse Dictionary Learning in Mechanistic Interpretability

**论文链接:** [http://arxiv.org/abs/2512.05534v1](http://arxiv.org/abs/2512.05534v1)

**作者:** Yiming Tang, Harshvardhan Saini, Yizhen Liao, Dianbo Liu

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种针对稀疏字典学习方法的统一理论框架，解释了这些方法在实践中有效的原因，并为一些经验观察到的现象提供了理论解释。

### 背景

随着AI模型在各个领域展现出显著能力，理解它们学习到的表示以及如何处理信息对于科学进步和可信部署变得越来越重要。机制可解释性研究表明，神经网络将有意义的概念表示为表示空间中的方向，并以叠加方式编码多个概念。稀疏字典学习方法通过训练具有稀疏约束的辅助模型来解耦这些叠加概念，但这些方法缺乏充分的理论基础。

### 目的

开发第一个将稀疏字典学习视为统一优化问题的理论框架，为广泛SDL方法提供形式化基础，解释一些经验观察到的现象，并通过控制实验验证理论结果。

### 方法

开发统一理论框架，将SDL视为一个统一的优化问题，展示不同方法如何实例化该框架，并对优化景观进行严格分析。

### 主要发现

为一些经验观察到的现象提供了首个理论解释，包括特征吸收、死亡神经元和神经元重采样技术，并设计了控制实验验证理论结果。

### 结论

该工作填补了SDL方法理论理解的空白，为理解和改进这些方法提供了坚实基础。

### 翻译

随着AI模型在各个领域取得显著能力，理解它们学习到的表示以及如何处理信息对于科学进步和可信部署变得越来越重要。最近在机制可解释性方面的工作表明，神经网络将有意义的概念表示为其表示空间中的方向，并且通常以叠加方式编码许多概念。各种稀疏字典学习方法，包括稀疏自编码器、转码器和交叉编码器，通过训练具有稀疏约束的辅助模型来解耦这些叠加的概念为可解释的特征。这些方法已经显示出显著的实证成功，但理论理解有限。现有的理论工作仅限于具有权重约束的稀疏自编码器，使得更广泛的SDL方法缺乏形式化基础。在这项工作中，我们开发了第一个将SDL视为统一优化问题的统一理论框架。我们展示了不同方法如何实例化理论框架，并对优化景观进行了严格分析。我们为一些经验观察到的现象提供了第一个理论解释，包括特征吸收、死亡神经元和神经元重采样技术。我们进一步设计了控制实验来验证我们的理论结果。


### 论文摘要

As AI models achieve remarkable capabilities across diverse domains, understanding what representations they learn and how they process information has become increasingly important for both scientific progress and trustworthy deployment. Recent works in mechanistic interpretability have shown that neural networks represent meaningful concepts as directions in their representation spaces and often encode many concepts in superposition. Various sparse dictionary learning (SDL) methods, including sparse autoencoders, transcoders, and crosscoders, address this by training auxiliary models with sparsity constraints to disentangle these superposed concepts into interpretable features. These methods have demonstrated remarkable empirical success but have limited theoretical understanding. Existing theoretical work is limited to sparse autoencoders with tied-weight constraints, leaving the broader family of SDL methods without formal grounding. In this work, we develop the first unified theoretical framework considering SDL as one unified optimization problem. We demonstrate how diverse methods instantiate the theoretical framwork and provide rigorous analysis on the optimization landscape. We provide the first theoretical explanations for some empirically observed phenomena, including feature absorption, dead neurons, and the neuron resampling technique. We further design controlled experiments to validate our theoretical results.

---

## 76. IDK-S: Incremental Distributional Kernel for Streaming Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2512.05531v1](http://arxiv.org/abs/2512.05531v1)

**作者:** Yang Xu, Yixiao Ma, Kaifeng Zhang, Zuliang Yang, Kai Ming Ting

**发布时间:** 2025-12-05

### GPT解析

### 总结

IDK-S是一种创新的增量分布核方法，通过结合隔离分布核的优势和轻量级增量更新机制，实现了高效且准确的异常检测。

### 背景

数据流异常检测面临重大挑战，需要在不断变化的分布中保持高检测精度的同时确保实时效率。

### 目的

介绍一种新的增量分布核方法，用于解决数据流异常检测的挑战。

### 方法

提出了一种名为IDK-S（增量分布核用于流异常检测）的新方法。该方法在核均值嵌入框架中创建了一种新的动态表示。该方法有两个关键创新：继承了隔离分布核的优势，采用轻量级增量更新机制显著降低计算开销。

### 主要发现

在十三个基准测试上的广泛实验表明，IDK-S在检测精度上优于现有最先进的方法，并且在运行速度上快得多，在许多情况下快一个数量级。

### 结论

IDK-S方法能够有效解决数据流异常检测的挑战，在保持高检测精度的同时实现实时效率。

### 翻译

数据流异常检测带来了重大挑战，需要方法在不断变化的分布中保持高检测精度的同时确保实时效率。在这里，我们引入IDK-S，一种用于流异常检测的新型增量分布核，它通过在核均值嵌入框架中创建新的动态表示，有效解决了这些挑战。IDK-S的优越性归因于两个关键创新。首先，它继承了隔离分布核的优势，这是一种离线检测器，由于使用数据相关核，比基础方法如隔离森林和局部离群因子显示出显著的性能优势。其次，它采用轻量级增量更新机制，与执行完全模型重新训练的简单基线策略相比，显著降低了计算开销。这是在不牺牲检测精度的情况下实现的，这一主张得到了其与完全重新训练模型的统计等价性的支持。我们在十三个基准测试上的广泛实验表明，IDK-S在检测精度上优于现有最先进的方法，同时运行速度显著更快，在许多情况下快一个数量级。


### 论文摘要

Anomaly detection on data streams presents significant challenges, requiring methods to maintain high detection accuracy among evolving distributions while ensuring real-time efficiency. Here we introduce $\mathcal{IDK}$-$\mathcal{S}$, a novel $\mathbf{I}$ncremental $\mathbf{D}$istributional $\mathbf{K}$ernel for $\mathbf{S}$treaming anomaly detection that effectively addresses these challenges by creating a new dynamic representation in the kernel mean embedding framework. The superiority of $\mathcal{IDK}$-$\mathcal{S}$ is attributed to two key innovations. First, it inherits the strengths of the Isolation Distributional Kernel, an offline detector that has demonstrated significant performance advantages over foundational methods like Isolation Forest and Local Outlier Factor due to the use of a data-dependent kernel. Second, it adopts a lightweight incremental update mechanism that significantly reduces computational overhead compared to the naive baseline strategy of performing a full model retraining. This is achieved without compromising detection accuracy, a claim supported by its statistical equivalence to the full retrained model. Our extensive experiments on thirteen benchmarks demonstrate that $\mathcal{IDK}$-$\mathcal{S}$ achieves superior detection accuracy while operating substantially faster, in many cases by an order of magnitude, than existing state-of-the-art methods.

---

## 77. See in Depth: Training-Free Surgical Scene Segmentation with Monocular Depth Priors

**论文链接:** [http://arxiv.org/abs/2512.05529v1](http://arxiv.org/abs/2512.05529v1)

**作者:** Kunyi Yang, Qingyu Wang, Cheng Yuan, Yutong Ban

**发布时间:** 2025-12-05

**备注:** The first two authors contributed equally

### GPT解析

### 总结

DepSeg是一种无需训练的框架，利用单目深度作为几何先验结合预训练视觉基础模型，实现腹腔镜场景的高效分割，显著减少了标注需求。

### 背景

腹腔镜场景的逐像素分割对计算机辅助手术至关重要，但由于密集注释的高成本而难以规模化。

### 目的

开发一种注释高效的分割方法，降低对大量标注数据的依赖。

### 方法

DepSeg首先估计相对深度图，提出深度引导的点提示，由SAM2转换为类无关掩码；然后通过池化预训练视觉特征和模板匹配进行掩码分类，构建模板库用于分类。

### 主要发现

在CholecSeg8k数据集上，DepSeg的mIoU达到35.9%，显著优于直接SAM2自动分割基线的14.7%；即使仅使用10-20%的目标模板也能保持有竞争力的性能。

### 结论

深度引导提示和基于模板的分类提供了一种注释高效的分割方法，在计算机辅助手术领域具有应用潜力。

### 翻译

腹腔镜场景的逐像素分割对计算机辅助手术至关重要，但由于密集标注的高成本而难以规模化。我们提出了深度引导的手术场景分割(DepSeg)，这是一种无需训练的框架，利用单目深度作为几何先验，结合预训练的视觉基础模型。DepSeg首先使用预训练的单目深度估计网络估计相对深度图，并提出深度引导的点提示，这些提示由SAM2转换为类无关掩码。每个掩码由池化的预训练视觉特征描述，并通过与从标注帧构建的模板库进行模板匹配进行分类。在CholecSeg8k数据集上，DepSeg比直接SAM2自动分割基线(35.9%对14.7% mIoU)有所改进，即使只使用10-20%的目标模板也能保持有竞争力的性能。这些结果表明，深度引导提示和基于模板的分类提供了一种标注高效的分割方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决腹腔镜手术场景的像素级分割问题，由于获取密集标注的高成本，这一任务难以扩展。这一问题在现实中非常重要，因为准确的手术场景分割对于器械跟踪、工作流程分析和手术室中的智能辅助至关重要，但传统方法需要大量专家标注，限制了临床应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到腹腔镜图像中器械通常比周围组织更靠近相机的几何特性，结合了现有预训练模型的优势：使用SAM2生成高质量掩码，DINOv3提供视觉特征描述，DepthAnythingV2提供深度估计。他们没有从零开始设计，而是巧妙整合这些现有工作，利用深度信息作为几何先验来引导分割过程，减少对密集标注的依赖。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用单目深度作为几何先验来引导分割，结合预训练的视觉基础模型实现无需训练的分割。整体流程包括：1)模板注册阶段：用少量标注构建特征模板库；2)深度引导点提示：从深度图生成高质量点提示；3)掩码提案：用SAM2将点提示转换为掩码；4)模板匹配：用DINOv3特征和模板库对掩码进行分类，生成最终分割结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)无需训练的分割框架，仅依赖少量标注构建模板库；2)深度引导提示策略，利用单目深度作为几何先验；3)实用设计优势，易于扩展且无需微调。相比传统完全监督方法，它减少了标注需求；相比直接使用SAM2的方法，它通过深度提示和模板匹配实现了语义分类；相比其他深度分割方法，它更适合腹腔镜手术场景的特定空间布局。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DepSeg通过结合单目深度先验和预训练视觉基础模型，提出了一种无需训练的腹腔镜手术场景分割方法，显著减少了标注需求，同时保持了与完全监督方法相竞争的性能。'}


### 论文摘要

Pixel-wise segmentation of laparoscopic scenes is essential for computer-assisted surgery but difficult to scale due to the high cost of dense annotations. We propose depth-guided surgical scene segmentation (DepSeg), a training-free framework that utilizes monocular depth as a geometric prior together with pretrained vision foundation models. DepSeg first estimates a relative depth map with a pretrained monocular depth estimation network and proposes depth-guided point prompts, which SAM2 converts into class-agnostic masks. Each mask is then described by a pooled pretrained visual feature and classified via template matching against a template bank built from annotated frames. On the CholecSeg8k dataset, DepSeg improves over a direct SAM2 auto segmentation baseline (35.9% vs. 14.7% mIoU) and maintains competitive performance even when using only 10--20% of the object templates. These results show that depth-guided prompting and template-based classification offer an annotation-efficient segmentation approach.

---

## 78. SCAIL: Towards Studio-Grade Character Animation via In-Context Learning of 3D-Consistent Pose Representations

**论文链接:** [http://arxiv.org/abs/2512.05905v1](http://arxiv.org/abs/2512.05905v1)

**作者:** Wenhao Yan, Sheng Ye, Zhuoyi Yang, Jiayan Teng, ZhenHui Dong, Kairui Wen, Xiaotao Gu, Yong-Jin Liu, Jie Tang

**发布时间:** 2025-12-05

### GPT解析

### 总结

SCAIL是一个工作室级别的角色动画框架，通过两个关键创新解决了现有方法在复杂场景中无法保持结构保真度和时间一致性的问题。

### 背景

尽管最近在角色动画方面取得了进展，但要达到工作室级别的生产标准仍然具有挑战性。现有方法可以将运动从驱动视频传输到参考图像，但在涉及复杂运动和跨身份动画的野外场景中往往表现不佳。

### 目的

开发SCAIL框架，解决角色动画在复杂场景中的结构保真度和时间一致性问题，达到工作室级别的生产标准。

### 方法

提出两种创新：1) 新的3D姿态表示，提供更强大和灵活的运动信号；2) 在扩散Transformer架构中引入全上下文姿态注入机制，实现完整的时空推理。同时开发精心策划的数据管道确保多样性和质量，并建立全面的基准系统进行评估。

### 主要发现

SCAIL实现了最先进的性能，显著提升了角色动画的可靠性和现实性，使其更接近工作室级别的生产标准。

### 结论

SCAIL框架通过创新的3D姿态表示和全上下文姿态注入机制，有效解决了复杂场景中的角色动画挑战，推动了角色动画技术的发展。

### 翻译

尽管最近取得了进展，但要达到工作室级别的角色动画制作标准仍然具有挑战性。现有方法可以将运动从驱动视频传输到参考图像，但在涉及复杂运动和跨身份动画的野外场景中，往往无法保持结构保真度和时间一致性。在这项工作中，我们提出了SCAIL（通过上下文学习的工作室级别角色动画），这是一个通过两个关键创新来解决这些挑战的框架。首先，我们提出了一种新的3D姿态表示，提供更强大和灵活的运动信号。其次，我们在扩散Transformer架构中引入了全上下文姿态注入机制，能够对完整运动序列进行有效的时空推理。为了符合工作室级别的要求，我们开发了一个精心策划的数据管道，确保多样性和质量，并建立了全面的基准进行系统评估。实验表明，SCAIL实现了最先进的性能，并将角色动画推向工作室级别的可靠性和现实性。


### 论文摘要

Achieving character animation that meets studio-grade production standards remains challenging despite recent progress. Existing approaches can transfer motion from a driving video to a reference image, but often fail to preserve structural fidelity and temporal consistency in wild scenarios involving complex motion and cross-identity animations. In this work, we present \textbf{SCAIL} (\textbf{S}tudio-grade \textbf{C}haracter \textbf{A}nimation via \textbf{I}n-context \textbf{L}earning), a framework designed to address these challenges from two key innovations. First, we propose a novel 3D pose representation, providing a more robust and flexible motion signal. Second, we introduce a full-context pose injection mechanism within a diffusion-transformer architecture, enabling effective spatio-temporal reasoning over full motion sequences. To align with studio-level requirements, we develop a curated data pipeline ensuring both diversity and quality, and establish a comprehensive benchmark for systematic evaluation. Experiments show that \textbf{SCAIL} achieves state-of-the-art performance and advances character animation toward studio-grade reliability and realism.

---

## 79. Predicting Price Movements in High-Frequency Financial Data with Spiking Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.05868v1](http://arxiv.org/abs/2512.05868v1)

**作者:** Brian Ezinwoke, Oliver Rhodes

**发布时间:** 2025-12-05

**备注:** 9 pages, 5 figures, 8 tables

### GPT解析

### 总结

该研究探索了脉冲神经网络(SNNs)在高频交易价格尖峰预测中的应用，通过贝叶斯优化进行超参数调优，比较了三种SNN架构，并使用惩罚性尖峰准确率(PSA)作为优化目标。

### 背景

现代高频交易环境以突然的价格波动为特征，既带来风险也带来机会，但传统金融模型无法捕捉所需的精细时间结构。脉冲神经网络因其自然处理离散事件和保持毫秒级时间的能力，适合应对这些挑战。

### 目的

研究旨在探索SNNs在高频价格尖峰预测中的应用，通过贝叶斯优化进行稳健的超参数调优来增强性能。

### 方法

将高频股票数据转换为脉冲序列，评估三种SNN架构：无监督STDP训练的SNN、具有显性抑制性竞争的新型SNN、监督反向传播网络。使用惩罚性尖峰准确率(PSA)作为贝叶斯优化的目标函数，确保预测价格尖峰率与实证率一致。

### 主要发现

模拟交易表明，使用PSA优化的模型持续优于使用尖峰准确率(SA)调整的模型和基准。扩展SNN模型与PSA结合在回测中实现了76.8%的最高累计回报率，显著优于监督模型的42.54%回报率。

### 结论

脉冲网络在稳健使用任务特定目标函数进行调优后，对于高频交易中有效的价格尖峰预测具有潜力。

### 翻译

现代高频交易(HFT)环境的特点是突然的价格波动，这既带来风险也带来机会，但传统的金融模型往往无法捕捉所需的精细时间结构。脉冲神经网络(SNNs)提供了一种生物启发的框架，非常适合这些挑战，因为它们自然具有处理离散事件和保持毫秒级时间的能力。这项工作研究了SNNs在高频价格尖峰预测中的应用，通过使用贝叶斯优化(BO)进行稳健的超参数调优来增强性能。这项工作将高频股票数据转换为脉冲序列，并评估了三种架构：一种已建立的无监督STDP训练的SNN，一种具有显性抑制性竞争的新型SNN，以及一种监督反向传播网络。贝叶斯优化的驱动是一个新颖的目标——惩罚性尖峰准确率(PSA)，旨在确保网络预测的价格尖峰率与价格事件的实证率保持一致。模拟交易表明，使用PSA优化的模型持续优于使用尖峰准确率(SA)调整的对应模型和基准。具体而言，使用PSA的扩展SNN模型在简单回测中实现了最高的累计回报率(76.8%)，显著优于监督替代模型(42.54%的回报率)。这些结果验证了脉冲网络在稳健使用任务特定目标函数进行调优后，对于高频交易中有效的价格尖峰预测具有潜力。


### 论文摘要

Modern high-frequency trading (HFT) environments are characterized by sudden price spikes that present both risk and opportunity, but conventional financial models often fail to capture the required fine temporal structure. Spiking Neural Networks (SNNs) offer a biologically inspired framework well-suited to these challenges due to their natural ability to process discrete events and preserve millisecond-scale timing. This work investigates the application of SNNs to high-frequency price-spike forecasting, enhancing performance via robust hyperparameter tuning with Bayesian Optimization (BO). This work converts high-frequency stock data into spike trains and evaluates three architectures: an established unsupervised STDP-trained SNN, a novel SNN with explicit inhibitory competition, and a supervised backpropagation network. BO was driven by a novel objective, Penalized Spike Accuracy (PSA), designed to ensure a network's predicted price spike rate aligns with the empirical rate of price events. Simulated trading demonstrated that models optimized with PSA consistently outperformed their Spike Accuracy (SA)-tuned counterparts and baselines. Specifically, the extended SNN model with PSA achieved the highest cumulative return (76.8%) in simple backtesting, significantly surpassing the supervised alternative (42.54% return). These results validate the potential of spiking networks, when robustly tuned with task-specific objectives, for effective price spike forecasting in HFT.

---

## 80. Phase-OTDR Event Detection Using Image-Based Data Transformation and Deep Learning

**论文链接:** [http://arxiv.org/abs/2512.05830v1](http://arxiv.org/abs/2512.05830v1)

**作者:** Muhammet Cagri Yeke, Samil Sirin, Kivilcim Yuksel, Abdurrahman Gumus

**发布时间:** 2025-12-05

**备注:** 22 pages, 11 figures, 5 tables

### GPT解析

### 总结

本研究提出了一种基于图像的光纤事件检测方法，通过将1D数据转换为灰度图像并组合成多通道RGB表示，使用迁移学习模型实现了高准确率的事件分类。

### 背景

光纤事件检测需要更有效的数据分析方法，特别是使用Phase-OTDR系统对六种事件进行分类。

### 目的

提高光纤事件检测的准确性和效率，同时减少数据集大小。

### 方法

将1D数据通过Gramian Angular Difference Field、Gramian Angular Summation Field和Recurrence Plot技术转换为灰度图像，然后组合成多通道RGB表示，使用迁移学习模型进行分析。

### 主要发现

EfficientNetB0模型实现98.84%分类准确率，DenseNet121模型实现98.24%分类准确率；5折交叉验证确认模型可靠性，测试准确率分别为99.07%和98.68%；方法能减少数据集大小并提高分析效率。

### 结论

基于图像的分析方法在解释复杂光纤传感数据方面具有变革潜力，为光纤监控系统的准确性和可靠性提供了显著改进；代码和数据集已在GitHub上公开。

### 翻译

本研究专注于光纤中的事件检测，特别是使用Phase-OTDR系统对六种事件进行分类。引入了一种新方法，通过将1D数据转换为灰度图像来增强Phase-OTDR数据分析，使用的技术包括Gramian Angular Difference Field、Gramian Angular Summation Field和Recurrence Plot。这些灰度图像被组合成多通道RGB表示，使迁移学习模型能够进行更强大和适应性强的分析。所提出的方法使用EfficientNetB0和DenseNet121模型分别实现了98.84%和98.24%的高分类准确率。5折交叉验证过程确认了这些模型的可靠性，测试准确率分别为99.07%和98.68%。使用公开可用的Phase-OTDR数据集，该研究展示了一种理解光纤事件的高效方法，同时减少了数据集大小并提高了分析效率。结果突显了基于图像的分析在解释复杂光纤传感数据方面的变革潜力，为光纤监控系统的准确性和可靠性提供了显著进步。代码和相应的基于图像的数据集已在GitHub上公开，以支持进一步研究：https://github.com/miralab-ai/Phase-OTDR-event-detection。


### 论文摘要

This study focuses on event detection in optical fibers, specifically classifying six events using the Phase-OTDR system. A novel approach is introduced to enhance Phase-OTDR data analysis by transforming 1D data into grayscale images through techniques such as Gramian Angular Difference Field, Gramian Angular Summation Field, and Recurrence Plot. These grayscale images are combined into a multi-channel RGB representation, enabling more robust and adaptable analysis using transfer learning models. The proposed methodology achieves high classification accuracies of 98.84% and 98.24% with the EfficientNetB0 and DenseNet121 models, respectively. A 5-fold cross-validation process confirms the reliability of these models, with test accuracy rates of 99.07% and 98.68%. Using a publicly available Phase-OTDR dataset, the study demonstrates an efficient approach to understanding optical fiber events while reducing dataset size and improving analysis efficiency. The results highlight the transformative potential of image-based analysis in interpreting complex fiber optic sensing data, offering significant advancements in the accuracy and reliability of fiber optic monitoring systems. The codes and the corresponding image-based dataset are made publicly available on GitHub to support further research: https://github.com/miralab-ai/Phase-OTDR-event-detection.

---

## 81. Machine-learning-enabled interpretation of tribological deformation patterns in large-scale MD data

**论文链接:** [http://arxiv.org/abs/2512.05818v1](http://arxiv.org/abs/2512.05818v1)

**作者:** Hendrik J. Ehrich, Marvin C. May, Stefan J. Eder

**发布时间:** 2025-12-05

**备注:** 19 pages, 11 figures

### GPT解析

### 总结

该研究介绍了一种数据驱动的工作流程，利用机器学习来自动化解释分子动力学模拟产生的摩擦学变形图案数据，实现了约96%的预测准确率，为减少大规模模拟需求提供了可能。

### 背景

分子动力学模拟已成为探索原子尺度摩擦学变形模式不可或缺的工具，但将高维数据转换为可解释的变形图谱仍是一个资源密集且 largely 手动的过程。

### 目的

开发一种自动化方法，用于从结构图像中识别和分类摩擦学变形特征，实现完全自动化、数据驱动的摩擦学机制图谱构建。

### 方法

1) 使用自编码器将含晶粒取向色的计算断层图像压缩为32维全局特征向量；2) 保留微观结构特征如晶界、堆垛层错、孪晶和部分晶格旋转；3) 结合学习表示与模拟元数据训练CNN-MLP模型；4) 采用排除训练区域的评估策略提高泛化能力。

### 主要发现

1) 强压缩后重建图像仍保留基本微观结构特征；2) 模型在验证数据上实现约96%预测准确率；3) 排除训练区域的评估策略提供更稳健的泛化度量；4) 可用机器学习自动识别和分类摩擦学变形特征。

### 结论

该研究证明了一种概念验证，表明可以使用机器学习从结构图像中自动识别和分类摩擦学变形特征，朝着完全自动化、数据驱动的摩擦学机制图谱构建迈出了第一步。

### 翻译

分子动力学模拟已成为探索原子尺度摩擦学变形模式不可或缺的工具。然而，将产生的高维数据转换为可解释的变形图谱仍然是一个资源密集且 largely 手动的过程。在这项工作中，我们引入了一种数据驱动的工作流程，利用无监督和监督学习来自动化这一解释步骤。从CuNi合金模拟中获得的含晶粒取向色的计算断层图像首先通过自编码器压缩为32维全局特征向量。尽管进行了这种强压缩，重建的图像仍保留了基本的微观结构特征：晶界、堆垛层错、孪晶和部分晶格旋转，仅省略了最精细的缺陷。然后，将学习到的表示与模拟元数据（成分、载荷、时间和温度以及空间位置）结合，以训练CNN-MLP模型预测主导变形模式。该模型在验证数据上实现了约96%的预测准确率。一种精细的评估策略，即将包含不同晶粒的整个空间区域排除在训练之外，提供了更稳健的泛化度量。该方法表明，可以使用机器学习从结构图像中自动识别和分类摩擦学变形特征。这个概念验证朝着完全自动化、数据驱动的摩擦学机制图谱构建迈出了第一步，并最终可能减少对大规模分子动力学模拟的需求。


### 论文摘要

Molecular dynamics (MD) simulations have become indispensable for exploring tribological deformation patterns at the atomic scale. However, transforming the resulting high-dimensional data into interpretable deformation pattern maps remains a resource-intensive and largely manual process. In this work, we introduce a data-driven workflow that automates this interpretation step using unsupervised and supervised learning. Grain-orientation-colored computational tomograph pictures obtained from CuNi alloy simulations were first compressed through an autoencoder to a 32-dimensional global feature vector. Despite this strong compression, the reconstructed images retained the essential microstructural motifs: grain boundaries, stacking faults, twins, and partial lattice rotations, while omitting only the finest defects. The learned representations were then combined with simulation metadata (composition, load, time, temperature, and spatial position) to train a CNN-MLP model to predict the dominant deformation pattern. The resulting model achieves a prediction accuracy of approximately 96% on validation data. A refined evaluation strategy, in which an entire spatial region containing distinct grains was excluded from training, provides a more robust measure of generalization. The approach demonstrates that essential tribological deformation signatures can be automatically identified and classified from structural images using Machine Learning. This proof of concept constitutes a first step towards fully automated, data-driven construction of tribological mechanism maps and, ultimately, toward predictive modeling frameworks that may reduce the need for large-scale MD simulation campaigns.

---

## 82. Over-the-Air Semantic Alignment with Stacked Intelligent Metasurfaces

**论文链接:** [http://arxiv.org/abs/2512.05657v1](http://arxiv.org/abs/2512.05657v1)

**作者:** Mario Edoardo Pandolfo, Kyriakos Stylianopoulos, George C. Alexandropoulos, Paolo Di Lorenzo

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种基于堆叠智能超表面(SIM)的空中语义对齐框架，直接在波域中实现潜在空间对齐，减少设备计算负担，有效处理异构模型间的语义不匹配问题。

### 背景

语义通信系统旨在在具备人工智能功能的设备间传输任务相关信息，但当异构发射器-接收器模型产生不匹配的潜在表示时，系统性能会下降。现有的语义对齐方法通常依赖发射器或接收器上的额外数字处理，增加了整体设备复杂度。

### 目的

开发一种无需额外数字处理的空中语义对齐方法，减少设备计算负担，实现异构模型间的语义对齐。

### 方法

提出基于堆叠智能超表面(SIM)的框架，将SIM建模为可训练的线性算子，能够模拟监督线性对齐器和零样本Parseval帧均衡器。开发基于梯度的优化程序，定制超表面传递函数以实现所需的语义映射。

### 主要发现

SIM能够准确重现监督和零样本语义均衡器，在高信噪比下达到90%的任务准确率，即使在低信噪比条件下也保持强大的鲁棒性。

### 结论

基于SIM的空中语义对齐框架为语义通信系统提供了一种高效、低复杂度的解决方案，能够在波域中直接实现潜在空间对齐，显著减少设备层面的计算负担。

### 翻译

语义通信系统旨在在具备人工智能功能的设备间传输任务相关信息，但当异构发射器-接收器模型产生不匹配的潜在表示时，系统性能会下降。现有的语义对齐方法通常依赖发射器或接收器上的额外数字处理，增加了整体设备复杂度。在本研究中，我们引入了首个基于堆叠智能超表面(SIM)的空中语义对齐框架，该框架能够在波域中直接实现潜在空间对齐，显著减少设备层面的计算负担。我们将SIM建模为可训练的线性算子，能够模拟监督线性对齐器和零样本Parseval帧均衡器。为了物理实现这些算子，我们开发了一种基于梯度的优化程序，定制超表面传递函数以实现所需的语义映射。使用异构视觉变换器(ViT)编码器的实验表明，SIM能够准确重现监督和零样本语义均衡器，在高信噪比(SNR)条件下达到90%的任务准确率，即使在低SNR值下也保持强大的鲁棒性。


### 论文摘要

Semantic communication systems aim to transmit task-relevant information between devices capable of artificial intelligence, but their performance can degrade when heterogeneous transmitter-receiver models produce misaligned latent representations. Existing semantic alignment methods typically rely on additional digital processing at the transmitter or receiver, increasing overall device complexity. In this work, we introduce the first over-the-air semantic alignment framework based on stacked intelligent metasurfaces (SIM), which enables latent-space alignment directly in the wave domain, reducing substantially the computational burden at the device level. We model SIMs as trainable linear operators capable of emulating both supervised linear aligners and zero-shot Parseval-frame-based equalizers. To realize these operators physically, we develop a gradient-based optimization procedure that tailors the metasurface transfer function to a desired semantic mapping. Experiments with heterogeneous vision transformer (ViT) encoders show that SIMs can accurately reproduce both supervised and zero-shot semantic equalizers, achieving up to 90% task accuracy in regimes with high signal-to-noise ratio (SNR), while maintaining strong robustness even at low SNR values.

---

## 83. Experts-Guided Unbalanced Optimal Transport for ISP Learning from Unpaired and/or Paired Data

**论文链接:** [http://arxiv.org/abs/2512.05635v1](http://arxiv.org/abs/2512.05635v1)

**作者:** Georgy Perevozchikov, Nancy Mehta, Egor Ershov, Radu Timofte

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种基于最优传输的无监督训练框架，能够在成对和非成对模式下训练任意ISP架构，减少了大规模成对数据的依赖，同时实现了与原始方法相当或更好的性能。

### 背景

现有的学习型图像信号处理(ISP)流水线虽然具有强大的端到端性能，但严重依赖于大规模的成对raw-to-sRGB数据集。获取这些成对数据的成本高昂，成为显著瓶颈。

### 目的

提出一种新颖的无监督训练框架，基于最优传输(Optimal Transport)，能够在成对和非成对模式下训练任意ISP架构，解决对大规模成对数据的依赖问题。

### 方法

首次成功应用非平衡最优传输(UOT)进行跨域翻译任务；UOT框架对目标sRGB数据中的异常值具有鲁棒性；引入'专家鉴别器委员会'作为混合对抗正则化器，提供专门梯度纠正特定ISP故障模式，包括颜色保真度、结构伪影和频域真实性。

### 主要发现

在成对模式下训练时，框架在所有指标上超过了原始成对方法的性能；在非成对模式下，框架在定量和定性性能上与原始成对训练的对手相媲美，甚至在某些情况下超越了它们。

### 结论

提出的框架能够有效减少对成对数据的依赖，同时保持或提高ISP流水线的性能，为ISP训练提供了新思路。

### 翻译

学习的图像信号处理(ISP)流水线提供了强大的端到端性能，但严重依赖于大规模的成对raw-to-sRGB数据集。这种对难以获取的成对数据的依赖仍然是一个重大瓶颈。为了应对这一挑战，我们引入了一种新颖的无监督训练框架，基于最优传输，能够在成对和非成对模式下训练任意ISP架构。我们是第一个成功应用非平衡最优传输(UOT)处理这种复杂跨域翻译任务的。基于UOT的框架对目标sRGB数据中的异常值具有鲁棒性，能够忽略那些映射成本极高的非典型样本。我们框架的一个关键组成部分是新颖的'专家鉴别器委员会'，这是一种混合对抗正则化器。该委员会通过提供专门的、有针对性的梯度来指导最优传输映射，以纠正特定的ISP故障模式，包括颜色保真度、结构伪影和频域真实性。为了证明我们方法的优越性，我们使用成对和非成对设置重新训练了现有的最先进ISP架构。实验表明，虽然我们的框架在成对模式下训练时在所有指标上都超过了原始成对方法的性能，但我们的非成对模式同时实现了与原始成对训练对手相当甚至在某些情况下超越的定量和定性性能。代码和预训练模型可在以下网址获取：https://github.com/gosha20777/EGUOT-ISP.git。


### 论文摘要

Learned Image Signal Processing (ISP) pipelines offer powerful end-to-end performance but are critically dependent on large-scale paired raw-to-sRGB datasets. This reliance on costly-to-acquire paired data remains a significant bottleneck. To address this challenge, we introduce a novel, unsupervised training framework based on Optimal Transport capable of training arbitrary ISP architectures in both unpaired and paired modes. We are the first to successfully apply Unbalanced Optimal Transport (UOT) for this complex, cross-domain translation task. Our UOT-based framework provides robustness to outliers in the target sRGB data, allowing it to discount atypical samples that would be prohibitively costly to map. A key component of our framework is a novel ``committee of expert discriminators,'' a hybrid adversarial regularizer. This committee guides the optimal transport mapping by providing specialized, targeted gradients to correct specific ISP failure modes, including color fidelity, structural artifacts, and frequency-domain realism. To demonstrate the superiority of our approach, we retrained existing state-of-the-art ISP architectures using our paired and unpaired setups. Our experiments show that while our framework, when trained in paired mode, exceeds the performance of the original paired methods across all metrics, our unpaired mode concurrently achieves quantitative and qualitative performance that rivals, and in some cases surpasses, the original paired-trained counterparts. The code and pre-trained models are available at: https://github.com/gosha20777/EGUOT-ISP.git.

---

## 84. Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales

**论文链接:** [http://arxiv.org/abs/2512.05620v1](http://arxiv.org/abs/2512.05620v1)

**作者:** Shikai Qiu, Zixi Chen, Hoang Phan, Qi Lei, Andrew Gordon Wilson

**发布时间:** 2025-12-05

**备注:** NeurIPS 2025. Code available at: https://github.com/charliezchen/scaling-matrix-preconditioning

### GPT解析

### 总结

本研究探讨了利用矩阵预处理技术的深度学习优化器在大规模模型中的有效性，通过超参数传输方法提升优化器性能。

### 背景

最近引入的利用矩阵预处理技术的深度学习优化器相对于AdamW优化器在小规模实验中显示出速度提升，但验证和复制这些成功结果的报告结果不一致。

### 目的

更好地理解预处理优化器在大规模情况下的有效性，研究如何通过超参数传输来扩展这些优化器。

### 方法

基于μP等先前工作，研究学习率和权重衰减如何随模型宽度和深度缩放，分析Shampoo、SOAP和Muon等多种优化器，并考虑blocking和grafting等常用技术的影响。

### 主要发现

根据μP缩放学习率可提高传输效果，但仍存在有限宽度偏差，可通过blocking和显式谱归一化缓解；权重衰减按1/宽度缩放对大多数优化器接近最优；应用正确缩放规则后，Muon和Shampoo在训练Llama架构语言模型时比AdamW分别快1.4倍和1.3倍，但不正确缩放会使速度提升随规模迅速消失。

### 结论

研究最优超参数传输对于在给定实际调整预算的情况下可靠地比较大规模优化器至关重要。

### 翻译

最近引入的几种利用矩阵预处理的深度学习优化器相对于当前主流的AdamW优化器显示出有希望的速度提升，尤其是在相对小规模的实验中。然而，验证和复制这些成功结果的报告显示结果不一致。为了更好地理解这些优化器在大规模情况下的有效性，本研究探讨了如何通过超参数传输来扩展预处理优化器，基于之前如μP的工作。我们研究了学习率和权重 decay应如何随模型宽度和深度缩放，以适应广泛的优化器，包括Shampoo、SOAP和Muon，同时考虑了blocking和grafting等常用技术的影响。我们发现，根据μP缩放学习率可以提高传输效果，但仍可能遭受显著的有限宽度偏差，导致最优学习率漂移，这可以通过blocking和显式谱归一化来缓解。对于计算最优缩放，我们发现权重衰减按1/宽度独立缩放在所有优化器中几乎是最佳的。应用这些缩放规则后，我们证明Muon和Shampoo在训练从190M到1.4B不等的Llama架构语言模型时，分别比AdamW快1.4倍和1.3倍，而在不正确的缩放下，这种速度提升会随着规模迅速消失。基于这些结果和进一步的消融实验，我们认为研究最优超参数传输对于在给定现实调整预算的情况下可靠地比较大规模优化器是必不可少的。


### 论文摘要

Several recently introduced deep learning optimizers utilizing matrix-level preconditioning have shown promising speedups relative to the current dominant optimizer AdamW, particularly in relatively small-scale experiments. However, efforts to validate and replicate their successes have reported mixed results. To better understand the effectiveness of these optimizers at scale, in this work we investigate how to scale preconditioned optimizers via hyperparameter transfer, building on prior works such as $μ$P. We study how the optimal learning rate and weight decay should scale with model width and depth for a wide range of optimizers, including Shampoo, SOAP, and Muon, accounting for the impact of commonly used techniques such as blocking and grafting. We find that scaling the learning rate according to $μ$P improves transfer, but can still suffer from significant finite-width deviations that cause drifting optimal learning rates, which we show can be mitigated by blocking and explicit spectral normalization. For compute-optimal scaling, we find scaling independent weight decay as $1/\mathrm{width}$ is nearly optimal across optimizers. Applying these scaling rules, we show Muon and Shampoo consistently achieve $1.4\times$ and $1.3\times$ speedup over AdamW for training Llama-architecture language models of sizes ranging from $190$M to $1.4$B, whereas the speedup vanishes rapidly with scale under incorrect scaling. Based on these results and further ablations, we argue that studying optimal hyperparameter transfer is essential for reliably comparing optimizers at scale given a realistic tuning budget.

---

## 85. Learning High-Fidelity Cloth Animation via Skinning-Free Image Transfer

**论文链接:** [http://arxiv.org/abs/2512.05593v1](http://arxiv.org/abs/2512.05593v1)

**作者:** Rong Wang, Wei Mao, Changsheng Lu, Hongdong Li

**发布时间:** 2025-12-05

**备注:** Accepted to 3DV 2026

### GPT解析

### 总结

本文提出了一种新颖的无蒙皮方法，通过独立估计顶点位置和顶点法线来生成3D服装变形，解决了现有方法中形状不对齐的问题，并通过2D图像传输和多模态融合实现了高质量的3D服装变形效果。

### 背景

生成3D服装变形对于虚拟试衣和扩展现实应用至关重要，但现有方法主要依赖线性混合蒙皮获取低频率姿势服装形状，仅回归高频褶皱，常导致形状不对齐和褶皱恢复不精确的问题。

### 目的

解决现有基于蒙皮的方法中形状不对齐导致的高频信号破坏和高保真褶皱恢复失败的问题，实现更高质量的3D服装变形生成。

### 方法

提出无蒙皮方法，独立估计顶点位置(低频率姿势服装形状)和顶点法线(高频局部褶皱细节)；将顶点属性编码为渲染纹理图像，通过2D图像传输实现3D服装变形；采用多模态融合结合两种频率模态的约束，从传输图像中恢复变形的3D服装。

### 主要发现

独立处理低频率和高频率模态可以有效避免形状不对齐问题；通过2D图像传输和预训练图像模型可以恢复更精细的褶皱细节；无需手动UV分区即可处理多样化拓扑结构的服装。

### 结论

所提出的方法在多种服装类型上显著提高了动画质量，并能比现有最先进的方法恢复更精细的褶皱细节。

### 翻译

我们提出了一种新颖的方法，可以从给定的身体姿势生成3D服装变形，这对于虚拟试衣和扩展现实等多种应用至关重要。为简化布料动力学，现有方法大多依赖线性混合蒙皮来获取低频率姿势服装形状，仅回归高频褶皱。然而，由于缺乏明确的蒙皮监督，这种基于蒙皮的方法在摆弄服装时常常产生形状不对齐，从而破坏高频信号并无法恢复高保真褶皱。为解决这一问题，我们提出了一种无蒙皮方法，独立估计姿势的顶点位置(用于低频率姿势服装形状)和顶点法线(用于高频局部褶皱细节)。这样，每种频率模态可以有效解耦，并由变形服装的几何形状直接监督。为进一步提高动画的视觉质量，我们提出将两种顶点属性编码为渲染的纹理图像，使3D服装变形可以通过2D图像传输等效实现。这使得我们能够利用强大的预训练图像模型来恢复褶皱中的细粒度视觉细节，同时保持对多样化拓扑结构服装的优异扩展性，无需依赖手动UV分区。最后，我们提出多模态融合，结合两种频率模态的约束，从传输的图像中稳健地恢复变形的3D服装。大量实验表明，我们的方法在多种服装类型上显著提高了动画质量，并比最先进的方法恢复了更精细的褶皱。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从底层人体姿势生成高质量的3D服装变形问题。现有方法依赖线性混合蒙皮技术获取低频服装形状并回归高频褶皱，但缺乏明确蒙皮监督，导致形状错位和褶皱细节丢失。这个问题在虚拟试衣、扩展现实和数字人等领域至关重要，因为高质量的服装变形直接影响用户体验的真实感和沉浸感。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别了现有蒙皮方法的局限性：缺乏明确监督导致错位和伪影。他们提出不依赖蒙皮的新思路，通过独立估计顶点位置(低频)和顶点法线(高频)来分解频率模态。作者借鉴了图像处理技术，将3D变形转化为2D图像传递任务，利用预训练图像模型恢复细节。还借鉴了多模态融合技术，结合两种模态的信息。整体设计受到视觉Transformer(DINO)和图像表示方法的启发，但创新性地应用于服装变形领域。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将服装变形分解为低频(顶点位置)和高频(顶点法线)两种模态，通过2D图像传递和多模态融合实现高质量变形。整体流程：1) 图像渲染：将3D服装和人体从多视角渲染为位置和法线图像；2) 2D图像传递：使用位置传递网络生成姿势变换后的位置图像，法线传递网络生成褶皱细节的法线图像；3) 3D多模态融合：从位置图像初始化顶点位置，通过优化融合法线信息恢复高频褶皱，并应用边缘长度、法线一致性和碰撞避免等约束提高质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 无蒙皮方法：直接估计顶点位置和法线，避免蒙皮错位；2) 2D图像传递：将3D变形转化为2D任务，利用预训练图像模型处理细节；3) 多模态融合：结合位置和法线信息，稳健恢复变形。相比之前工作的不同：1) 避免了基于蒙皮方法的错位和伪影问题；2) 无需手动UV映射，可扩展到各种拓扑结构；3) 利用图像模型的感知能力更好地捕捉细粒度细节；4) 通过多模态融合提高鲁棒性，特别是对不可见区域的处理。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种无蒙皮的3D服装变形方法，通过将变形分解为低频位置和高频法线两种模态，并利用2D图像传递和多模态融合技术，实现了高质量、高保真度的服装动画，显著优于现有方法。'}


### 论文摘要

We present a novel method for generating 3D garment deformations from given body poses, which is key to a wide range of applications, including virtual try-on and extended reality. To simplify the cloth dynamics, existing methods mostly rely on linear blend skinning to obtain low-frequency posed garment shape and only regress high-frequency wrinkles. However, due to the lack of explicit skinning supervision, such skinning-based approach often produces misaligned shapes when posing the garment, consequently corrupts the high-frequency signals and fails to recover high-fidelity wrinkles. To tackle this issue, we propose a skinning-free approach by independently estimating posed (i) vertex position for low-frequency posed garment shape, and (ii) vertex normal for high-frequency local wrinkle details. In this way, each frequency modality can be effectively decoupled and directly supervised by the geometry of the deformed garment. To further improve the visual quality of animation, we propose to encode both vertex attributes as rendered texture images, so that 3D garment deformation can be equivalently achieved via 2D image transfer. This enables us to leverage powerful pretrained image models to recover fine-grained visual details in wrinkles, while maintaining superior scalability for garments of diverse topologies without relying on manual UV partition. Finally, we propose a multimodal fusion to incorporate constraints from both frequency modalities and robustly recover deformed 3D garments from transferred images. Extensive experiments show that our method significantly improves animation quality on various garment types and recovers finer wrinkles than state-of-the-art methods.

---

## 86. Poodle: Seamlessly Scaling Down Large Language Models with Just-in-Time Model Replacement

**论文链接:** [http://arxiv.org/abs/2512.05525v1](http://arxiv.org/abs/2512.05525v1)

**作者:** Nils Strassenburg, Boris Glavic, Tilmann Rabl

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种即时模型替换(JITR)方法，在使用大型语言模型时自动替换为针对特定任务更高效的模型，从而保留LLMs的易用性同时节省成本和能源。

### 背景

企业越来越多地依赖大型语言模型(LLMs)来自动化简单重复任务，而不是开发自定义机器学习模型。LLMs几乎不需要训练示例，且无需专业知识即可使用。然而，与小型模型相比，LLMs需要更多的资源和能源消耗，而这些小型模型在简单任务中通常能达到相似的预测性能。

### 目的

提出一种JITR方法，在识别到对LLM的重复调用时，透明地将模型替换为针对该特定性能良好的更便宜的替代品，从而保留LLMs的易用性和低开发努力，同时节省大量成本和能源。

### 方法

讨论实现JITR面临的主要挑战，包括识别重复任务和创建自定义模型。特别指出模型搜索和迁移学习将在JITR中发挥关键作用，以有效识别和微调重复任务的模型。使用名为Poodle的JITR原型进行实验。

### 主要发现

通过JITR原型Poodle，在示例任务上实现了显著的节省。

### 结论

JITR方法能够在保持LLMs易用性的同时，显著降低成本和能源消耗，特别适合处理简单重复任务。

### 翻译

企业越来越多地依赖大型语言模型(LLMs)来自动化简单重复任务，而不是开发自定义机器学习模型。LLMs几乎不需要训练示例，且无需专业知识即可使用。然而，与小型模型相比，LLMs需要更多的资源和能源消耗，而这些小型模型在简单任务中通常能达到相似的预测性能。在本文中，我们提出了即时模型替换(JITR)的愿景，即在识别到对LLM的重复调用时，将模型透明地替换为针对该特定性能良好的更便宜的替代品。JITR保留了LLMs的易用性和低开发努力，同时节省大量成本和能源。我们讨论了实现这一愿景面临的主要挑战，包括识别重复任务和创建自定义模型。具体来说，我们认为模型搜索和迁移学习将在JITR中发挥关键作用，以有效识别和微调重复任务的模型。通过我们的JITR原型Poodle，我们在示例任务上实现了显著的节省。


### 论文摘要

Businesses increasingly rely on large language models (LLMs) to automate simple repetitive tasks instead of developing custom machine learning models. LLMs require few, if any, training examples and can be utilized by users without expertise in model development. However, this comes at the cost of substantially higher resource and energy consumption compared to smaller models, which often achieve similar predictive performance for simple tasks. In this paper, we present our vision for just-in-time model replacement (JITR), where, upon identifying a recurring task in calls to an LLM, the model is replaced transparently with a cheaper alternative that performs well for this specific task. JITR retains the ease of use and low development effort of LLMs, while saving significant cost and energy. We discuss the main challenges in realizing our vision regarding the identification of recurring tasks and the creation of a custom model. Specifically, we argue that model search and transfer learning will play a crucial role in JITR to efficiently identify and fine-tune models for a recurring task. Using our JITR prototype Poodle, we achieve significant savings for exemplary tasks.

---

## 87. Beyond Adam: Disentangling Optimizer Effects in the Fine-Tuning of Atomistic Foundation Models

**论文链接:** [http://arxiv.org/abs/2512.05489v1](http://arxiv.org/abs/2512.05489v1)

**作者:** Xiaoqing Liu, Yangshuai Wang, Teng Zhao

**发布时间:** 2025-12-05

### GPT解析

### 总结

本研究评估了七种优化算法在原子基础模型微调中的性能，发现AdamW和ScheduleFree表现最佳，而SGD表现较差，并提出了通过二阶精炼提升模型性能的方法。

### 背景

原子基础模型在计算材料科学中构成了一种范式转变，提供了具有广泛化学空间可转移性的通用机器学习原子间势能函数。然而，为适应这些预训练模型到特定目标系统而进行的微调过程中，优化算法的影响尚未得到充分表征。

### 目的

对不同优化算法在基础模型微调过程中的性能进行严格基准测试，评估它们对能量和力精度的影响，以及对下游物理性质的影响。

### 方法

对七种一阶优化器（包括Adam、AdamW、RAdam、SGD、LAMB、Ranger和ScheduleFree）进行基准测试，用于分子、晶体和液体体系的基础模型微调。基于分布内和分布外构型的能量和力精度，以及对下游物理性质（如弹性模量、声子谱和界面动力学）的影响来评估这些算法。通过预处理框架将每个优化器视为梯度的数据相关线性变换来解释结果。

### 主要发现

AdamW和ScheduleFree在所有体系中都表现出优异的曲率条件和力精度；随机梯度下降(SGD)表现出缓慢收敛和不稳定性；简短的二阶精炼阶段可以减少损失景观中的残余各向异性，提高物理观测的保真度，而不会增加推理成本。

### 结论

这些发现为选择和设计优化器提供了概念见解和实际指导，以确保通用原子间势能的稳定和高效微调。

### 翻译

原子基础模型通过提供具有广泛化学空间可转移性的通用机器学习原子间势能函数，在计算材料科学中构成了一种范式转变。虽然微调对于将这些预训练模型适应到特定目标系统至关重要，但优化算法对此过程的影响仍未得到充分表征。在本工作中，我们对七种一阶优化器（包括Adam、AdamW、RAdam、SGD、LAMB、Ranger和ScheduleFree）进行了严格的基准测试，用于分子、晶体和液体体系的基础模型微调。我们基于分布内和分布外构型的能量和力精度，以及它们对下游物理性质（如弹性模量、声子谱和界面动力学）的影响来评估这些算法。我们通过预处理框架解释这些经验结果，该框架将每个优化器视为梯度的数据相关线性变换。这种分析阐明了不同的更新规则如何对有效损失Hessian施加特定的谱滤波器。在所有体系中，AdamW和ScheduleFree实现了优异的曲率条件和力精度，而随机梯度下降则表现出缓慢收敛和不稳定性。此外，我们证明简短二阶精炼阶段可以减少损失景观中的残余各向异性，提高物理观测的保真度，而不会增加推理成本。这些发现为选择和设计优化器提供了概念见解和实际指导，以确保通用原子间势能的稳定和高效微调。


### 论文摘要

Atomistic foundation models constitute a paradigm shift in computational materials science by providing universal machine-learned interatomic potentials with broad transferability across chemical spaces. Although fine-tuning is essential for adapting these pretrained models to specific target systems, the influence of the optimization algorithm on this process remains insufficiently characterized. In this work, we perform a rigorous benchmark of seven first-order optimizers, including Adam, AdamW, RAdam, SGD, LAMB, Ranger, and ScheduleFree, for the fine-tuning of foundation models across molecular, crystalline, and liquid regimes. We evaluate these algorithms based on energy and force accuracy for both in-distribution and out-of-distribution configurations, as well as their impact on downstream physical properties such as elastic moduli, phonon spectra, and interfacial dynamics. We interpret these empirical results through a preconditioning framework that views each optimizer as a data-dependent linear transformation of the gradient. This analysis clarifies how different update rules impose specific spectral filters on the effective loss Hessian. Across all regimes, AdamW and ScheduleFree achieve superior curvature conditioning and force accuracy, whereas stochastic gradient descent exhibits slow convergence and instability. Furthermore, we demonstrate that a brief second-order refinement stage reduces residual anisotropy in the loss landscape and enhances the fidelity of physical observables without increasing inference costs. These findings provide conceptual insight and practical guidance for selecting and designing optimizers to ensure the stable and efficient fine-tuning of universal interatomic potentials.

---

## 88. IdealTSF: Can Non-Ideal Data Contribute to Enhancing the Performance of Time Series Forecasting Models?

**论文链接:** [http://arxiv.org/abs/2512.05442v1](http://arxiv.org/abs/2512.05442v1)

**作者:** Hua Wang, Jinghao Lu, Fan Zhang

**发布时间:** 2025-12-05

**备注:** Accepted at AAAI 2026

### GPT解析

### 总结

该研究提出IdealTSF框架，通过同时利用理想正样本和非理想负样本（如缺失值和异常值）来提高时间序列预测性能，包含预训练、训练和优化三个步骤。

### 背景

深度学习在时间序列预测中表现优异，但序列数据中的缺失值和异常值等问题限制了其进一步发展。先前研究主要关注特征提取或将次优数据作为知识转移的正样本。

### 目的

利用非理想负样本来增强事件预测，提出一种更有效的时间序列预测方法。

### 方法

提出IdealTSF框架，包含三个步骤：1)预训练：从负样本数据中提取知识预训练模型；2)训练：将序列数据转换为理想正样本；3)优化：应用带有对抗性扰动的负优化机制。

### 主要发现

实验表明，负样本数据在基础注意力架构中为时间序列预测释放了显著潜力。

### 结论

IdealTSF特别适合处理含有噪声样本或低质量数据的应用场景。

### 翻译

深度学习在时间序列预测任务中表现出强大的性能。然而，序列数据中的缺失值和异常值等问题阻碍了其在预测任务中的进一步发展。先前的研究主要关注从序列数据中提取特征信息或将这些次优数据作为知识转移的正样本。更有效的方法是利用这些非理想负样本来增强事件预测。为此，本研究强调了非理想负样本的优势，并提出了IdealTSF框架，该框架整合了理想正样本和负样本用于时间序列预测。IdealTSF包含三个渐进步骤：预训练、训练和优化。它首先通过从负样本数据中提取知识来预训练模型，然后在训练过程中将序列数据转换为理想正样本。此外，还应用了带有对抗性扰动的负优化机制。大量实验表明，负样本数据在基础注意力架构中为时间序列预测释放了显著潜力。因此，IdealTSF特别适合应用于含有噪声样本或低质量数据的场景。


### 论文摘要

Deep learning has shown strong performance in time series forecasting tasks. However, issues such as missing values and anomalies in sequential data hinder its further development in prediction tasks. Previous research has primarily focused on extracting feature information from sequence data or addressing these suboptimal data as positive samples for knowledge transfer. A more effective approach would be to leverage these non-ideal negative samples to enhance event prediction. In response, this study highlights the advantages of non-ideal negative samples and proposes the IdealTSF framework, which integrates both ideal positive and negative samples for time series forecasting. IdealTSF consists of three progressive steps: pretraining, training, and optimization. It first pretrains the model by extracting knowledge from negative sample data, then transforms the sequence data into ideal positive samples during training. Additionally, a negative optimization mechanism with adversarial disturbances is applied. Extensive experiments demonstrate that negative sample data unlocks significant potential within the basic attention architecture for time series forecasting. Therefore, IdealTSF is particularly well-suited for applications with noisy samples or low-quality data.

---

## 89. RevoNAD: Reflective Evolutionary Exploration for Neural Architecture Design

**论文链接:** [http://arxiv.org/abs/2512.05403v1](http://arxiv.org/abs/2512.05403v1)

**作者:** Gyusam Chang, Jeongyoon Yoon, Shin han yi, JaeHyeok Lee, Sujin Jang, Sangpil Kim

**发布时间:** 2025-12-05

### GPT解析

### 总结

RevoNAD是一种反思性进化编排器，有效连接基于大语言模型的推理与反馈对齐的架构搜索，在多个数据集上实现了最先进的神经架构设计性能。

### 背景

大语言模型的进步使神经架构设计系统能生成不受预定义搜索空间限制的新架构，但基于LLM的生成面临挑战：标记级设计循环是离散且不可微的，阻碍了反馈平滑指导架构改进，导致模式崩溃或漂移到不可行设计。

### 目的

引入RevoNAD，一种反思性进化编排器，有效桥接基于LLM的推理与反馈对齐的架构搜索，解决现有方法的局限性。

### 方法

1) 多轮多专家共识：将孤立设计规则转化为有意义的架构线索；2) 自适应反思探索：利用奖励方差调整探索程度，在反馈不确定时探索，稳定性达到时细化；3) 帕累托引导的进化选择：促进同时优化准确性、效率、延迟、置信度和结构多样性的架构。

### 主要发现

在CIFAR10、CIFAR100、ImageNet16-120、COCO-5K和Cityscape等数据集上，RevoNAD实现了最先进的性能，消融和迁移研究验证了其在实际可靠且可部署的神经架构设计中的有效性。

### 结论

RevoNAD成功解决了LLM驱动神经架构设计中的关键挑战，实现了高性能且可实际部署的神经架构设计。

### 翻译

最近利用大型语言模型（LLMs）的进展使得神经架构设计（NAD）系统能够生成不受手动预定义搜索空间限制的新架构。然而，基于LLM的生成仍然面临挑战：标记级设计循环是离散且不可微的，这阻碍了反馈平滑指导架构改进。这些方法通常会出现模式崩溃，陷入冗余结构，或者当建设性推理基础不牢固时漂移到不可行的设计。我们引入了RevoNAD，一个反思性进化编排器，有效桥接基于LLM的推理与反馈对齐的架构搜索。首先，RevoNAD提出多轮多专家共识，将孤立的设计规则转化为有意义的架构线索。然后，自适应反思探索利用奖励方差调整探索程度；在反馈不确定时探索，在达到稳定性时细化。最后，帕累托引导的进化选择有效促进同时优化准确性、效率、延迟、置信度和结构多样性的架构。在CIFAR10、CIFAR100、ImageNet16-120、COCO-5K和Cityscape上，RevoNAD实现了最先进的性能。消融和迁移研究进一步验证了RevoNAD在允许实际可靠且可部署的神经架构设计方面的有效性。


### 论文摘要

Recent progress in leveraging large language models (LLMs) has enabled Neural Architecture Design (NAD) systems to generate new architecture not limited from manually predefined search space. Nevertheless, LLM-driven generation remains challenging: the token-level design loop is discrete and non-differentiable, preventing feedback from smoothly guiding architectural improvement. These methods, in turn, commonly suffer from mode collapse into redundant structures or drift toward infeasible designs when constructive reasoning is not well grounded. We introduce RevoNAD, a reflective evolutionary orchestrator that effectively bridges LLM-based reasoning with feedback-aligned architectural search. First, RevoNAD presents a Multi-round Multi-expert Consensus to transfer isolated design rules into meaningful architectural clues. Then, Adaptive Reflective Exploration adjusts the degree of exploration leveraging reward variance; it explores when feedback is uncertain and refines when stability is reached. Finally, Pareto-guided Evolutionary Selection effectively promotes architectures that jointly optimize accuracy, efficiency, latency, confidence, and structural diversity. Across CIFAR10, CIFAR100, ImageNet16-120, COCO-5K, and Cityscape, RevoNAD achieves state-of-the-art performance. Ablation and transfer studies further validate the effectiveness of RevoNAD in allowing practically reliable, and deployable neural architecture design.

---

## 90. State-Conditional Adversarial Learning: An Off-Policy Visual Domain Transfer Method for End-to-End Imitation Learning

**论文链接:** [http://arxiv.org/abs/2512.05335v1](http://arxiv.org/abs/2512.05335v1)

**作者:** Yuxiang Liu, Shengfan Cao

**发布时间:** 2025-12-05

### GPT解析

### 总结

该研究针对端到端模仿学习中的视觉域转移问题，提出了一种状态条件对抗学习(SCAL)框架，有效解决了在目标域数据严格离策略、无专家且稀缺情况下的域转移挑战。

### 背景

研究在现实且具有挑战性的环境中进行，目标域数据是严格离策略(off-policy)、无专家(expert-free)和稀缺的，这给域转移带来了困难。

### 目的

研究视觉域转移用于端到端模仿学习，解决在目标域数据有限且无专家指导情况下的学习问题。

### 方法

首先提供理论分析，表明目标域模仿损失可被源域损失加上状态条件潜在KL散度上界；基于此提出状态条件对抗学习(SCAL)框架，使用基于判别器的条件KL项估计器来对齐基于系统状态的潜在分布。

### 主要发现

在基于BARC-CARLA模拟器构建的视觉多样化的自动驾驶环境中的实验表明，SCAL实现了强大的迁移能力和高效的样本效率。

### 结论

SCAL方法能够在目标域数据稀缺且无专家指导的情况下有效进行视觉域转移，具有良好的样本效率，适合实际应用场景。

### 翻译

我们研究在端到端模仿学习中的视觉域转移，这是一个现实且具有挑战性的场景，其中目标域数据是严格离策略、无专家且稀缺的。我们首先提供了理论分析，表明目标域的模仿损失可以被源域损失加上源域和目标域观察模型之间的状态条件潜在KL散度所上界。受这一结果指导，我们提出了状态条件对抗学习，一个离策略对抗框架，它使用基于判别器的条件KL项估计器来对齐基于系统状态的潜在分布。在基于BARC-CARLA模拟器构建的视觉多样化的自动驾驶环境中的实验表明，SCAL实现了强大的迁移能力和高效的样本效率。


### 论文摘要

We study visual domain transfer for end-to-end imitation learning in a realistic and challenging setting where target-domain data are strictly off-policy, expert-free, and scarce. We first provide a theoretical analysis showing that the target-domain imitation loss can be upper bounded by the source-domain loss plus a state-conditional latent KL divergence between source and target observation models. Guided by this result, we propose State- Conditional Adversarial Learning, an off-policy adversarial framework that aligns latent distributions conditioned on system state using a discriminator-based estimator of the conditional KL term. Experiments on visually diverse autonomous driving environments built on the BARC-CARLA simulator demonstrate that SCAL achieves robust transfer and strong sample efficiency.

---

## 91. Variance Matters: Improving Domain Adaptation via Stratified Sampling

**论文链接:** [http://arxiv.org/abs/2512.05226v1](http://arxiv.org/abs/2512.05226v1)

**作者:** Andrea Napoli, Paul White

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了VaRDASS，一种专门用于无监督域适应的随机方差减少技术，通过分层采样方法解决域差异估计高方差问题，提高目标域性能。

### 背景

领域偏移是将机器学习模型部署到现实世界时的关键挑战，无监督域适应通过训练过程中最小化域差异来应对这一问题。

### 目的

开发一种专门针对无监督域适应的随机方差减少技术，解决域差异估计中的高方差问题。

### 方法

提出VaRDASS方法，考虑相关对齐和最大均值差异两种特定差异度量，设计专门的分层目标，提出期望和最坏情况误差边界，并证明MMD目标在特定假设下理论最优，同时引入并分析一个k-means风格优化算法。

### 主要发现

在三个域偏移数据集上的实验表明，VaRDASS提高了差异估计准确性和目标域性能。

### 结论

VaRDASS作为首个专门用于无监督域适应的方差减少技术，有效解决了传统方法中域差异估计的高方差问题，提升了模型在目标域的性能表现。

### 翻译

领域偏移是将机器学习模型部署到现实世界时面临的关键挑战。无监督域适应旨在通过训练过程中最小化域差异来解决这个问题，但在随机设置中，域差异的估计存在高方差问题，这可能阻碍该方法的理论优势。本文提出了VaRDASS（通过分层采样的方差减少域适应），这是第一个专门用于无监督域适应的随机方差减少技术。我们考虑了两种特定的差异度量——相关对齐和最大均值差异，并为这些术语设计了专门的分层目标。我们提出了期望和最坏情况误差边界，并证明在特定假设下，我们为MMD提出的理论上是最佳的（即最小化方差）的目标。最后，引入并分析了一个实用的k-means风格优化算法。在三个域偏移数据集上的实验表明，该方法提高了差异估计准确性和目标域性能。


### 论文摘要

Domain shift remains a key challenge in deploying machine learning models to the real world. Unsupervised domain adaptation (UDA) aims to address this by minimising domain discrepancy during training, but the discrepancy estimates suffer from high variance in stochastic settings, which can stifle the theoretical benefits of the method. This paper proposes Variance-Reduced Domain Adaptation via Stratified Sampling (VaRDASS), the first specialised stochastic variance reduction technique for UDA. We consider two specific discrepancy measures -- correlation alignment and the maximum mean discrepancy (MMD) -- and derive ad hoc stratification objectives for these terms. We then present expected and worst-case error bounds, and prove that our proposed objective for the MMD is theoretically optimal (i.e., minimises the variance) under certain assumptions. Finally, a practical k-means style optimisation algorithm is introduced and analysed. Experiments on three domain shift datasets demonstrate improved discrepancy estimation accuracy and target domain performance.

---

## 92. Advanced Unsupervised Learning: A Comprehensive Overview of Multi-View Clustering Techniques

**论文链接:** [http://arxiv.org/abs/2512.05169v1](http://arxiv.org/abs/2512.05169v1)

**作者:** Abdelmalik Moujahid, Fadi Dornaika

**发布时间:** 2025-12-04

**DOI:** 10.1007/s10462-025-11240-8

### GPT解析

### 总结

这是一篇关于多视图聚类(MVC)的综述性研究，系统性地分类了MVC方法，分析了其优缺点和实际挑战，并探讨了未来趋势和跨学科应用。

### 背景

机器学习技术面临诸多挑战，包括计算限制、单视图学习算法的局限性以及处理来自不同领域、来源或视图的大型数据集的复杂性。

### 目的

这篇综述有三个主要贡献：(1)系统地将多视图聚类方法分为明确的类别；(2)深入分析各种方法的优缺点和实际挑战；(3)前瞻性地讨论MVC研究中的新兴趋势、跨学科应用和未来方向。

### 方法

这是一项广泛的综述工作，包括回顾140多篇基础和最新出版物，开发关于集成策略(如早期融合、晚期融合和联合学习)的比较见解，以及结构化研究医疗保健、多媒体和社交网络分析等领域的实际用例。

### 主要发现

多视图聚类能够弥补单视图方法的不足，为各种无监督学习任务提供更丰富的数据表示和有效解决方案。尽管多视图数据具有固有复杂性，但其语义丰富的特性增加了其实用性。

### 结论

通过整合这些努力，这项工作旨在填补MVC研究中的现有空白，并为该领域的进步提供可行的见解。

### 翻译

机器学习技术面临诸多挑战才能实现最佳性能。这些挑战包括计算限制、单视图学习算法的局限性以及处理来自不同领域、来源或视图的大型数据集的复杂性。在这一背景下，多视图聚类(MVC)作为一种无监督多视图学习方法， emerges as a powerful approach to overcome these challenges. MVC弥补了单视图方法的缺点，为各种无监督学习任务提供了更丰富的数据表示和有效解决方案。与传统单视图方法相比，多视图数据语义丰富的特性增加了其实用性，尽管它具有固有的复杂性。这篇综述有三个方面的贡献：(1)将多视图聚类方法系统地分为明确的组别，包括协同训练、协同正则化、子空间、深度学习、基于核、基于锚点和基于图策略；(2)深入分析各自的优势、劣势和实际挑战，如可扩展性和不完整数据；(3)前瞻性地讨论MVC研究中的新兴趋势、跨学科应用和未来方向。这项研究代表了大量的工作，包括回顾140多篇基础和最新出版物，开发关于集成策略(如早期融合、晚期融合和联合学习)的比较见解，以及结构化研究医疗保健、多媒体和社交网络分析等领域的实际用例。通过整合这些努力，这项工作旨在填补MVC研究中的现有空白，并为该领域的进步提供可行的见解。


### 论文摘要

Machine learning techniques face numerous challenges to achieve optimal performance. These include computational constraints, the limitations of single-view learning algorithms and the complexity of processing large datasets from different domains, sources or views. In this context, multi-view clustering (MVC), a class of unsupervised multi-view learning, emerges as a powerful approach to overcome these challenges. MVC compensates for the shortcomings of single-view methods and provides a richer data representation and effective solutions for a variety of unsupervised learning tasks. In contrast to traditional single-view approaches, the semantically rich nature of multi-view data increases its practical utility despite its inherent complexity. This survey makes a threefold contribution: (1) a systematic categorization of multi-view clustering methods into well-defined groups, including co-training, co-regularization, subspace, deep learning, kernel-based, anchor-based, and graph-based strategies; (2) an in-depth analysis of their respective strengths, weaknesses, and practical challenges, such as scalability and incomplete data; and (3) a forward-looking discussion of emerging trends, interdisciplinary applications, and future directions in MVC research. This study represents an extensive workload, encompassing the review of over 140 foundational and recent publications, the development of comparative insights on integration strategies such as early fusion, late fusion, and joint learning, and the structured investigation of practical use cases in the areas of healthcare, multimedia, and social network analysis. By integrating these efforts, this work aims to fill existing gaps in MVC research and provide actionable insights for the advancement of the field.

---

## 93. How to Tame Your LLM: Semantic Collapse in Continuous Systems

**论文链接:** [http://arxiv.org/abs/2512.05162v1](http://arxiv.org/abs/2512.05162v1)

**作者:** C. M. Wyss

**发布时间:** 2025-12-04

**备注:** 35 pages, 1 figure. Exolytica AI Technical Report XTR-2025-01

### GPT解析

### 总结

论文提出了大型语言模型的语义动力学理论，将其形式化为连续状态机，证明了语义特征定理，解释了离散符号语义如何从连续计算中涌现。

### 背景

大型语言模型的语义动力学研究，需要理解其内部语义空间的演化机制。

### 目的

建立一种理论框架，形式化大型语言模型的语义动力学，并解释离散符号语义如何从连续计算中涌现。

### 方法

将大型语言模型形式化为连续状态机(CSMs)，使用转移算子P分析语义质量的传播，在正则性假设下证明语义特征定理(SCT)，并将其扩展到随机和非绝热设置。

### 主要发现

转移算子P在正则性假设下是紧致的且具有离散谱；主要特征函数诱导出有限数量的不变意义谱盆地；谱可聚集性和逻辑温和性是一致的；连续激活流形可坍缩为有限且可逻辑解释的本体论；缓慢漂移的核保持紧致性、谱相干性和盆地结构。

### 结论

大型语言模型的语义动力学可以通过连续状态机框架理解，离散符号语义可以从连续计算中自然涌现，这一理论框架适用于多种设置。

### 翻译

我们通过将大型语言模型形式化为连续状态机(CSMs)来发展语义动力学的一般理论：光滑动力系统，其潜在流形在概率转移算子下演化。相关的转移算子P编码了语义质量的传播。在适度的正则性假设下，P是紧致的且具有离散谱。在此框架中，我们证明了语义特征定理(SCT)：P的主要特征函数诱导出有限数量的不变意义谱盆地，每个谱盆地都在实数上的o-minimal结构中可定义。因此，谱可聚集性和逻辑温和性是一致的。这解释了离散符号语义如何从连续计算中出现：连续激活流形坍缩为有限且可逻辑解释的本体论。我们进一步将SCT扩展到随机和非绝热(时间非齐次)设置，表明缓慢漂移的核保持紧致性、谱相干性和盆地结构。


### 论文摘要

We develop a general theory of semantic dynamics for large language models by formalizing them as Continuous State Machines (CSMs): smooth dynamical systems whose latent manifolds evolve under probabilistic transition operators. The associated transfer operator $P: L^2(M,μ) \to L^2(M,μ)$ encodes the propagation of semantic mass. Under mild regularity assumptions (compactness, ergodicity, bounded Jacobian), $P$ is compact with discrete spectrum. Within this setting, we prove the Semantic Characterization Theorem (SCT): the leading eigenfunctions of $P$ induce finitely many spectral basins of invariant meaning, each definable in an o-minimal structure over $\mathbb{R}$. Thus spectral lumpability and logical tameness coincide. This explains how discrete symbolic semantics can emerge from continuous computation: the continuous activation manifold collapses into a finite, logically interpretable ontology. We further extend the SCT to stochastic and adiabatic (time-inhomogeneous) settings, showing that slowly drifting kernels preserve compactness, spectral coherence, and basin structure.

---

## 94. Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons and Manage Hallucinations

**论文链接:** [http://arxiv.org/abs/2512.05156v1](http://arxiv.org/abs/2512.05156v1)

**作者:** Igor Halperin

**发布时间:** 2025-12-04

**备注:** 23 pages, 6 figures

### GPT解析

### 总结

本研究提出两种新的无监督指标评估大型语言模型对任务的忠实性，基于信息论和热力学原理。将LLM建模为二部信息引擎，通过转换矩阵和KL散度量化忠实性，发现高忠实性通常与低熵产生相关。这些指标可用于LLM评估和幻觉控制。

### 背景

评估大型语言模型对给定任务的忠实性是一个复杂挑战。

### 目的

提出两种新的无监督指标用于忠实性评估，利用信息论和热力学的见解。

### 方法

将LLM视为二部信息引擎，隐藏层作为麦克斯韦妖控制上下文到答案的转换；将QCA三元组建模为共享主题上的概率分布；使用转换矩阵分别编码查询目标和实际结果；通过KL散度量化忠实性并进行凸优化；提出基于热力学的语义熵产生指标。

### 主要发现

高忠实性通常意味着低熵产生。

### 结论

SF和SEP指标可以联合或单独用于LLM评估和幻觉控制。

### 翻译

评估大型语言模型对给定任务的忠实性是一个复杂挑战。我们利用信息论和热力学的见解，提出了两种新的无监督指标用于忠实性评估。我们的方法将LLM视为一个二部信息引擎，其中隐藏层作为麦克斯韦妖，通过提示控制上下文到答案的转换。我们将问题-上下文-答案三元组建模为共享主题上的概率分布。从上下文到问题和答案的主题转换被建模为转换矩阵，分别编码查询目标和实际结果。我们的语义忠实性指标通过这些矩阵之间的KL散度来量化任何给定三元组的忠实性。通过对这个KL散度进行凸优化同时推断这两个矩阵，并通过将最小散度映射到单位区间获得最终指标，其中较高分数表示更高的忠实性。此外，我们提出了一种基于热力学的语义熵产生指标用于答案生成，并表明高忠实性通常意味着低熵产生。SF和SEP指标可以联合或单独用于LLM评估和幻觉控制。我们在LLM对公司SEC 10-K文件的摘要生成上展示了我们的框架。


### 论文摘要

Evaluating faithfulness of Large Language Models (LLMs) to a given task is a complex challenge. We propose two new unsupervised metrics for faithfulness evaluation using insights from information theory and thermodynamics. Our approach treats an LLM as a bipartite information engine where hidden layers act as a Maxwell demon controlling transformations of context $C $ into answer $A$ via prompt $Q$. We model Question-Context-Answer (QCA) triplets as probability distributions over shared topics. Topic transformations from $C$ to $Q$ and $A$ are modeled as transition matrices ${\bf Q}$ and ${\bf A}$ encoding the query goal and actual result, respectively. Our semantic faithfulness (SF) metric quantifies faithfulness for any given QCA triplet by the Kullback-Leibler (KL) divergence between these matrices. Both matrices are inferred simultaneously via convex optimization of this KL divergence, and the final SF metric is obtained by mapping the minimal divergence onto the unit interval [0,1], where higher scores indicate greater faithfulness. Furthermore, we propose a thermodynamics-based semantic entropy production (SEP) metric in answer generation, and show that high faithfulness generally implies low entropy production. The SF and SEP metrics can be used jointly or separately for LLM evaluation and hallucination control. We demonstrate our framework on LLM summarization of corporate SEC 10-K filings.

---

## 95. Label-Efficient Point Cloud Segmentation with Active Learning

**论文链接:** [http://arxiv.org/abs/2512.05759v1](http://arxiv.org/abs/2512.05759v1)

**作者:** Johannes Meyer, Jasper Hoffmann, Felix Schulz, Dominik Merkle, Daniel Buescher, Alexander Reiterer, Joschka Boedecker, Wolfram Burgard

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种用于3D点云语义分割的主动学习方法，通过2D网格分割和网络集成不确定性估计来选择最有价值的标注区域，在多个数据集上取得了优异的性能。

### 背景

3D点云数据的语义分割通常伴随着高昂的标注成本。主动学习可以自动化选择需要标注的数据，减少达到满意性能所需的标注总量。

### 目的

提出一种新颖且易于实现的策略，用于将点云分离为可标注区域，并确定下一个最有价值的标注数据。

### 方法

使用2D网格将点云细分为列，采用网络集成来估计网络输出中的不确定性，以确定下一个要标注的数据。在S3DIS数据集、Toronto-3D数据集以及部分手动标注的弗莱堡市大规模城市3D点云上评估该方法。

### 主要发现

该方法在所有数据集上表现与复杂的最先进方法相当甚至更好。研究结果表明，在点云背景下，标注面积可能是比标注点数量更有意义的主动学习算法度量标准。

### 结论

所提出的方法是一种有效且简单的点云主动学习策略，能够减少标注成本同时保持或提高性能。

### 翻译

3D点云数据的语义分割通常伴随着高昂的标注成本。主动学习可以自动化选择需要标注数据的过程，减少达到满意性能所需的标注总量。最近用于3D点云的主动学习方法通常基于复杂的启发式方法，将点云分割为可标注区域并选择对神经网络训练最有益的数据。在这项工作中，我们提出了一种新颖且易于实现的策略来分离点云为可标注区域。在我们的方法中，我们使用2D网格将点云细分为列。为了确定下一个要标注的数据，我们采用网络集成来估计网络输出的不确定性。我们在S3DIS数据集、Toronto-3D数据集以及我们部分手动标注的弗莱堡市大规模城市3D点云上评估了我们的方法。广泛的评估表明，我们的方法在所有数据集上的性能与复杂的最先进方法相当甚至更好。此外，我们提供的结果表明，在点云背景下，标注面积可能是比标注点数量更有意义的主动学习算法度量标准。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D点云数据语义分割的高标注成本问题。这个问题在现实中非常重要，因为语义分割是机器人、城市规划和环境监测等应用的关键技术，而城市点云数据的语义分割对于风、水和热模拟特别重要，可以帮助识别城市脆弱区域以增强对气候变化的抵御能力。然而，精确标注3D点云数据需要从多视角绘制大量多边形，过程极为昂贵和耗时。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者识别出3D点云主动学习面临两大挑战：区域分离（将点云分割成可标注区域）和区域选择（选择最有益区域进行标注）。他们观察到现有方法使用复杂启发式技术，需要繁琐预处理且可能导致难以标注的区域。作者借鉴了主动学习领域中基于深度集成的不确定性估计方法，但摒弃了复杂启发式技术，转而设计简单易用的方法：使用2D网格将点云分成列作为可标注区域，并采用网络集成估计不确定性进行区域选择。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用简单的空间列分割点云和基于集成的区域选择方法。整体流程包括：1)从初始标记数据集训练模型；2)使用2D网格将点云分成空间列；3)通过计算网络集成的不确定性（熵或变化率）选择最有信息的列；4)人类专家标注这些列；5)使用新标注数据重新训练模型；6)重复此过程直到达到期望性能。这种方法显著减少了预处理时间和计算复杂度。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)使用简单2D网格分割点云为空间列，替代复杂的监督体素方法；2)采用基于集成的标准（熵和变化率）进行区域选择，而非复杂混合启发式方法；3)提出使用标注面积而非标注点数作为更有效的度量标准；4)在大型城市3D点云数据集上验证方法。相比之前工作，该方法计算效率更高（预处理时间显著减少），实现更简单，性能相当或更好，且更符合人类标注的实际工作量。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种简单高效的主动学习方法，通过空间列分割和基于集成的区域选择，显著降低了3D点云语义分割的标注成本，同时达到了与或优于复杂最先进方法的性能。'}


### 论文摘要

Semantic segmentation of 3D point cloud data often comes with high annotation costs. Active learning automates the process of selecting which data to annotate, reducing the total amount of annotation needed to achieve satisfactory performance. Recent approaches to active learning for 3D point clouds are often based on sophisticated heuristics for both, splitting point clouds into annotatable regions and selecting the most beneficial for further neural network training. In this work, we propose a novel and easy-to-implement strategy to separate the point cloud into annotatable regions. In our approach, we utilize a 2D grid to subdivide the point cloud into columns. To identify the next data to be annotated, we employ a network ensemble to estimate the uncertainty in the network output. We evaluate our method on the S3DIS dataset, the Toronto-3D dataset, and a large-scale urban 3D point cloud of the city of Freiburg, which we labeled in parts manually. The extensive evaluation shows that our method yields performance on par with, or even better than, complex state-of-the-art methods on all datasets. Furthermore, we provide results suggesting that in the context of point clouds the annotated area can be a more meaningful measure for active learning algorithms than the number of annotated points.

---

## 96. NormalView: sensor-agnostic tree species classification from backpack and aerial lidar data using geometric projections

**论文链接:** [http://arxiv.org/abs/2512.05610v1](http://arxiv.org/abs/2512.05610v1)

**作者:** Juho Korkeala, Jesse Muhojoki, Josef Taher, Klaara Salolahti, Matti Hyyppä, Antero Kukko, Juha Hyyppä

**发布时间:** 2025-12-05

**备注:** 19 pages, 8 figures

### GPT解析

### 总结

本研究提出了一种名为NormalView的与传感器无关的基于投影的深度学习方法，用于从点云数据中分类树种。该方法将局部几何信息嵌入二维投影，并使用YOLOv11网络进行分类。实验表明，在MLS数据上达到95.5%的准确率，在ALS数据上达到91.8%的准确率。多光谱强度信息提高了分类性能。该方法与传感器无关，仅依赖几何信息，研究团队已公开发布相关数据集。

### 背景

激光扫描已被证明是评估森林环境分解的有价值工具，而移动激光扫描（MLS）在极其精确的树木级别清查方面显示出巨大潜力。

### 目的

提出NormalView，一种与传感器无关的基于投影的深度学习方法，用于从点云数据中分类树种。

### 方法

NormalView将局部几何信息嵌入到二维投影中，以法向量估计的形式，并使用这些投影作为图像分类网络YOLOv11的输入。研究还检查了多光谱辐射强度信息对分类性能的影响。模型在高密度MLS数据（7个物种，约5000个点/平方米）和高密度机载激光扫描（ALS）数据（9个物种，>1000个点/平方米）上进行了训练和测试。

### 主要发现

在MLS数据上，NormalView总体准确率（宏平均准确率）达到95.5%（94.8%），在ALS数据上达到91.8%（79.1%）。研究发现来自多个扫描仪的强度信息在树种分类中提供好处，在多光谱ALS数据集上，最佳模型是使用多光谱ALS所有三个通道强度信息的模型。

### 结论

基于投影的方法，当增强了几何信息并与最先进的图像分类骨干网络结合时，可以实现卓越的结果。这些方法与传感器无关，仅依赖于几何信息。研究团队已公开发布了研究中使用的MLS数据集。

### 翻译

激光扫描已被证明是评估森林环境分解的无价工具。移动激光扫描（MLS）在极其精确的树木级别清查方面显示出巨大潜力。在本研究中，我们提出了NormalView，一种与传感器无关的基于投影的深度学习方法，用于从点云数据中分类树种。NormalView将局部几何信息嵌入到二维投影中，以法向量估计的形式，并将这些投影用作图像分类网络YOLOv11的输入。此外，我们检查了多光谱辐射强度信息对分类性能的影响。我们在高密度MLS数据（7个物种，约5000个点/平方米）和高密度机载激光扫描（ALS）数据（9个物种，>1000个点/平方米）上训练和测试了我们的模型。在MLS数据上，NormalView实现了95.5%（94.8%）的总体准确率（宏平均准确率），在ALS数据上实现了91.8%（79.1%）。我们发现，来自多个扫描仪的强度信息在树种分类中提供好处，而在多光谱ALS数据集上，最佳模型是使用多光谱ALS所有三个通道强度信息的模型。这项研究表明，当增强了几何信息并与最先进的图像分类骨干网络结合时，基于投影的方法可以实现卓越的结果。关键是，这些方法与传感器无关，仅依赖于几何信息。此外，我们公开发布了研究中使用的MLS数据集。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决的问题是树木物种分类，特别是如何从不同类型的激光雷达数据（背包式移动激光扫描和航空激光扫描）中准确识别不同树种。这个问题在现实中非常重要，因为它有助于监测生物多样性、管理森林风险、了解森林生态系统的丰富性，并对森林资源管理和经济决策产生重要影响。特别是在北方森林中，准确识别关键物种（如白杨）对保护生物多样性热点至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者在思考过程中分析了现有方法的局限性：传统机器学习方法依赖手工特征，3D深度学习方法计算成本高，而投影方法可能未充分利用几何信息。他们借鉴了投影方法的基本思想、YOLOv11图像分类模型和法向量估计技术，但创新性地将法向量估计结果编码为RGB颜色，将3D几何信息嵌入2D投影图像中。作者还设计了多角度投影和切片图像以减少信息损失，并系统研究了多光谱信息的影响。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将3D点云数据转换为2D投影图像，并将每个点的局部几何信息（通过法向量估计）编码为颜色信息，然后利用先进的图像分类模型进行分类。实现流程包括：1)数据预处理（去噪、地面过滤、树木分割）；2)图像创建（多角度投影和切片图像，使用三种着色方案）；3)模型训练（使用YOLOv11架构和数据增强）；4)模型评估（使用多种评估指标）；5)推理（从多角度聚合预测结果）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)传感器无关的几何表示，仅依赖法向量信息；2)通过将法向量编码为RGB颜色，有效嵌入3D几何信息到2D图像；3)多角度投影和切片图像设计；4)系统研究多光谱信息对分类的影响。相比之前工作，它不依赖手工特征（与传统ML方法相比），利用了更先进的图像分类模型（与其他投影方法相比），计算效率更高（与3D深度学习方法相比），且同时适用于高密度和较低密度的激光扫描数据。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'NormalView通过将树木点云的局部几何信息编码为多视角投影图像，实现了传感器无关的高精度树木物种分类，同时证明了多光谱信息对提高分类性能，特别是对少数物种识别的重要性。'}


### 论文摘要

Laser scanning has proven to be an invaluable tool in assessing the decomposition of forest environments. Mobile laser scanning (MLS) has shown to be highly promising for extremely accurate, tree level inventory. In this study, we present NormalView, a sensor-agnostic projection-based deep learning method for classifying tree species from point cloud data. NormalView embeds local geometric information into two-dimensional projections, in the form of normal vector estimates, and uses the projections as inputs to an image classification network, YOLOv11. In addition, we inspected the effect of multispectral radiometric intensity information on classification performance. We trained and tested our model on high-density MLS data (7 species, ~5000 pts/m^2), as well as high-density airborne laser scanning (ALS) data (9 species, >1000 pts/m^2). On the MLS data, NormalView achieves an overall accuracy (macro-average accuracy) of 95.5 % (94.8 %), and 91.8 % (79.1 %) on the ALS data. We found that having intensity information from multiple scanners provides benefits in tree species classification, and the best model on the multispectral ALS dataset was a model using intensity information from all three channels of the multispectral ALS. This study demonstrates that projection-based methods, when enhanced with geometric information and coupled with state-of-the-art image classification backbones, can achieve exceptional results. Crucially, these methods are sensor-agnostic, relying only on geometric information. Additionally, we publically release the MLS dataset used in the study.

---

## 97. Efficient Text Classification with Conformal In-Context Learning

**论文链接:** [http://arxiv.org/abs/2512.05732v1](http://arxiv.org/abs/2512.05732v1)

**作者:** Ippokratis Pantelidis, Korbinian Randl, Aron Henriksson

**发布时间:** 2025-12-05

**备注:** 10 pages, 4 tables, 2 figures

### GPT解析

### 总结

CICLe是一种结合轻量级基础分类器和Conformal Prediction的资源高效框架，通过自适应减少候选类别集合来指导LLM提示，在多样化的NLP分类基准上表现出色，提高了数据与计算效率。

### 背景

大型语言模型(LLMs)展现出强大的上下文学习能力，但它们在文本分类中的有效性高度依赖于提示设计，并且计算成本很高。CICLe被提出作为一种资源高效的解决方案。

### 目的

系统探索CICLe在多个领域的适用性和效率优势，对CICLe在多样化的NLP分类基准上进行全面评估。

### 方法

CICLe结合了一个轻量级基础分类器和Conformal Prediction，通过自适应减少候选类别集合来指导LLM提示。研究者在多样化的NLP分类基准上对CICLe进行了全面评估。

### 主要发现

当基础分类器的训练样本量充足时，CICLe持续优于其基础分类器，并优于少样本提示基线；在低数据情况下，CICLe的表现具有可比性；CICLe将样本数量和提示长度分别减少了高达34.45%和25.16%；CICLe使得可以使用更小的模型同时保持有竞争力的性能；CICLe对于类别不平衡的文本分类任务特别有利。

### 结论

CICLe是一种实用且可扩展的高效文本分类方法，结合了传统分类器的稳健性和LLMs的适应性，在数据和计算效率方面取得了显著提升。

### 翻译

大型语言模型(LLMs)展现出强大的上下文学习能力，但它们在文本分类中的有效性高度依赖于提示设计，并且计算成本很高。一致性上下文学习(CICLe)已被提出一种资源高效框架，它将轻量级基础分类器与一致性预测相结合，通过自适应减少候选类别集合来指导LLM提示。然而，其在单一领域之外的更广泛适用性和效率优势尚未得到系统探索。在本文中，我们对CICLe在多样化的NLP分类基准上进行了全面评估。结果表明，当基础分类器的训练样本量充足时，CICLe持续优于其基础分类器，并优于少样本提示基线；在低数据情况下，CICLe的表现具有可比性。在效率方面，CICLe将样本数量和提示长度分别减少了高达34.45%和25.16%，并使得可以使用更小模型同时保持有竞争力的性能。此外，CICLe对于类别不平衡的文本分类任务特别有利。这些发现突显了CICLe作为一种实用且可扩展的高效文本分类方法，结合了传统分类器的稳健性和LLMs的适应性，并在数据和计算效率方面取得了显著提升。


### 论文摘要

Large Language Models (LLMs) demonstrate strong in-context learning abilities, yet their effectiveness in text classification depends heavily on prompt design and incurs substantial computational cost. Conformal In-Context Learning (CICLe) has been proposed as a resource-efficient framework that integrates a lightweight base classifier with Conformal Prediction to guide LLM prompting by adaptively reducing the set of candidate classes. However, its broader applicability and efficiency benefits beyond a single domain have not yet been systematically explored. In this paper, we present a comprehensive evaluation of CICLe across diverse NLP classification benchmarks. The results show that CICLe consistently improves over its base classifier and outperforms few-shot prompting baselines when the sample size is sufficient for training the base classifier, and performs comparably in low-data regimes. In terms of efficiency, CICLe reduces the number of shots and prompt length by up to 34.45% and 25.16%, respectively, and enables the use of smaller models with competitive performance. CICLe is furthermore particularly advantageous for text classification tasks with high class imbalance. These findings highlight CICLe as a practical and scalable approach for efficient text classification, combining the robustness of traditional classifiers with the adaptability of LLMs, and achieving substantial gains in data and computational efficiency.

---

## 98. Grounded Multilingual Medical Reasoning for Question Answering with Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.05658v1](http://arxiv.org/abs/2512.05658v1)

**作者:** Pietro Ferrazzi, Aitor Soroa, Rodrigo Agerri

**发布时间:** 2025-12-05

**备注:** Under Review

### GPT解析

### 总结

该研究提出了一种基于事实医学知识的多语言推理轨迹生成方法，在医疗问答任务中取得了显著成果。

### 背景

具有推理能力的大型语言模型在医疗问答领域展现出强大潜力，但现有方法主要关注英语，且依赖通用LLM的蒸馏，引发了对医疗知识可靠性的担忧。

### 目的

开发一种能够生成基于事实医学知识的多语言推理轨迹的方法，以提高医疗问答系统的可靠性和多语言支持能力。

### 方法

生成50万条英语、意大利语和西班牙语的推理轨迹，使用检索增强生成方法基于维基百科医学信息；扩展MedQA和MedMCQA数据集到意大利语和西班牙语；在领域内和领域外的医疗QA基准测试中评估方法效果。

### 主要发现

推理轨迹在使用上下文学习(少样本)和监督微调时都能提高性能，在80亿参数的LLMs中取得了最先进的结果。

### 结论

这些资源可以支持多语言环境下更安全、更透明的临床决策支持工具的开发。

### 翻译

研究团队发布了完整的资源套件，包括推理轨迹、翻译的QA数据集、医学维基百科和微调模型。


### 论文摘要

Large Language Models (LLMs) with reasoning capabilities have recently demonstrated strong potential in medical Question Answering (QA). Existing approaches are largely English-focused and primarily rely on distillation from general-purpose LLMs, raising concerns about the reliability of their medical knowledge. In this work, we present a method to generate multilingual reasoning traces grounded in factual medical knowledge. We produce 500k traces in English, Italian, and Spanish, using a retrievalaugmented generation approach over medical information from Wikipedia. The traces are generated to solve medical questions drawn from MedQA and MedMCQA, which we extend to Italian and Spanish. We test our pipeline in both in-domain and outof-domain settings across Medical QA benchmarks, and demonstrate that our reasoning traces improve performance both when utilized via in-context learning (few-shot) and supervised fine-tuning, yielding state-of-the-art results among 8B-parameter LLMs. We believe that these resources can support the development of safer, more transparent clinical decision-support tools in multilingual settings. We release the full suite of resources: reasoning traces, translated QA datasets, Medical-Wikipedia, and fine-tuned models.

---

## 99. Ontology Learning with LLMs: A Benchmark Study on Axiom Identification

**论文链接:** [http://arxiv.org/abs/2512.05594v1](http://arxiv.org/abs/2512.05594v1)

**作者:** Roos M. Bakker, Daan L. Di Scala, Maaike H. T. de Boer, Stephan A. Raaijmakers

**发布时间:** 2025-12-05

**备注:** Submitted to Semantic Web Journal, under review

### GPT解析

### 总结

本研究探讨了使用大型语言模型(LLMs)自动识别本体论公理(axioms)的能力，提出了一个名为OntoAxiom的基准测试，并评估了不同提示策略和模型性能。

### 背景

本体论是结构化领域知识的重要工具，但其开发需要大量建模和专业知识。近年来，随着自然语言处理技术和大型语言模型的发展，自动化本体学习取得了显著进展。

### 目的

研究旨在解决识别公理(定义类和属性间逻辑关系的基本本体组件)的挑战，通过系统测试LLMs在这一任务上的表现来评估其有效性。

### 方法

创建包含九个中等大小本体论的基准测试(共17,118个三元组和2,771个公理)，关注五种公理类型；评估12个LLMs使用两种提示策略：直接方法(一次查询所有公理)和逐公理方法(AbA)(每个提示查询一个公理)。

### 主要发现

AbA提示策略比直接方法产生更高的F1分数；不同公理类型的识别难度存在差异；领域影响性能(如FOAF本体子类公理得分为0.642，而音乐本体仅为0.218)；较大LLMs表现优于较小模型，但小模型在资源受限情况下可能仍有价值。

### 结论

尽管LLMs的整体性能不足以完全自动化公理识别，但它们可以为本体工程师提供有价值的候选公理，支持本体论的开发和完善。

### 翻译

本体论是结构化领域知识的重要工具，但其开发是一项复杂任务，需要大量的建模和领域专业知识。旨在自动化这一过程的本体学习在过去十年取得了进展，特别是随着自然语言处理技术的改进以及最近大型语言模型(LLMs)的增长。本文研究了识别公理的挑战：定义类和属性之间逻辑关系的基本本体论组件。在这项工作中，我们引入了一个本体论公理基准OntoAxiom，并在该基准上系统测试了LLMs的公理识别能力，评估了不同的提示策略、本体论和公理类型。该基准包含九个中等大小的本体论，共有17,118个三元组和2,771个公理。我们专注于子类、不相交、子属性、域和范围公理。为了评估LLM性能，我们比较了十二个LLMs在三种设置和两种提示策略下的表现：一种直接方法，我们一次性查询所有公理；另一种是逐公理(AbA)方法，每个提示只查询一个公理。我们的研究表明，AbA提示比直接方法产生更高的F1分数。然而，不同公理的性能各不相同，表明某些公理更具挑战性。领域也影响性能：FOAF本体在子类公理上得分为0.642，而音乐本体仅为0.218。较大的LLMs表现优于较小的模型，但较小的模型在资源受限的情况下可能仍然可行。尽管整体性能不足以完全自动化公理识别，但LLMs可以为本体工程师提供有价值的候选公理，支持本体论的开发和完善。


### 论文摘要

Ontologies are an important tool for structuring domain knowledge, but their development is a complex task that requires significant modelling and domain expertise. Ontology learning, aimed at automating this process, has seen advancements in the past decade with the improvement of Natural Language Processing techniques, and especially with the recent growth of Large Language Models (LLMs). This paper investigates the challenge of identifying axioms: fundamental ontology components that define logical relations between classes and properties. In this work, we introduce an Ontology Axiom Benchmark OntoAxiom, and systematically test LLMs on that benchmark for axiom identification, evaluating different prompting strategies, ontologies, and axiom types. The benchmark consists of nine medium-sized ontologies with together 17.118 triples, and 2.771 axioms. We focus on subclass, disjoint, subproperty, domain, and range axioms. To evaluate LLM performance, we compare twelve LLMs with three shot settings and two prompting strategies: a Direct approach where we query all axioms at once, versus an Axiom-by-Axiom (AbA) approach, where each prompt queries for one axiom only. Our findings show that the AbA prompting leads to higher F1 scores than the direct approach. However, performance varies across axioms, suggesting that certain axioms are more challenging to identify. The domain also influences performance: the FOAF ontology achieves a score of 0.642 for the subclass axiom, while the music ontology reaches only 0.218. Larger LLMs outperform smaller ones, but smaller models may still be viable for resource-constrained settings. Although performance overall is not high enough to fully automate axiom identification, LLMs can provide valuable candidate axioms to support ontology engineers with the development and refinement of ontologies.

---

## 100. TS-HINT: Enhancing Semiconductor Time Series Regression Using Attention Hints From Large Language Model Reasoning

**论文链接:** [http://arxiv.org/abs/2512.05419v1](http://arxiv.org/abs/2512.05419v1)

**作者:** Jonathan Adam Rico, Nagarajan Raghavan, Senthilnath Jayavelu

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了TS-Hint，一个时间序列基础模型框架，结合思维链推理，通过注意力机制数据和显著性数据提供注意力提示，解决了现有数据驱动方法在半导体制造过程中材料去除率估计中时间动力学损失和数据需求大的问题。

### 背景

现有数据驱动方法依赖从时间序列提取静态特征来近似半导体制造过程中的材料去除率，但这会导致时间动力学损失，且需要大量数据进行有效训练。

### 目的

开发一种能够在有限数据情况下有效学习时间序列特征并保持时间动力学信息的模型。

### 方法

提出TS-Hint，一个时间序列基础模型(TSFM)框架，集成思维链推理，在训练过程中基于注意力机制数据和显著性数据提供注意力提示。

### 主要发现

实验结果表明该模型在有限数据设置下通过少样本学习有效工作，能够直接从多元时间序列特征中学习。

### 结论

TS-Hint模型能够在数据有限的情况下有效学习时间序列特征，并保持时间动力学信息。

### 翻译

现有的数据驱动方法依赖于从时间序列中提取静态特征来近似半导体制造过程中的材料去除率(MRR)，如化学机械抛光(CMP)。然而，这会导致时间动力学的损失。此外，这些方法需要大量数据进行有效训练。在本文中，我们提出了TS-Hint，一个时间序列基础模型(TSFM)框架，集成了思维链推理，该推理在训练过程中基于注意力机制数据和显著性数据提供注意力提示。实验结果证明了我们的模型在有限数据设置下的有效性，通过少样本学习实现，并能直接从多元时间序列特征中学习。


### 论文摘要

Existing data-driven methods rely on the extraction of static features from time series to approximate the material removal rate (MRR) of semiconductor manufacturing processes such as chemical mechanical polishing (CMP). However, this leads to a loss of temporal dynamics. Moreover, these methods require a large amount of data for effective training. In this paper, we propose TS-Hint, a Time Series Foundation Model (TSFM) framework, integrated with chain-of-thought reasoning which provides attention hints during training based on attention mechanism data and saliency data. Experimental results demonstrate the effectiveness of our model in limited data settings via few-shot learning and can learn directly from multivariate time series features.

---

## 101. To Think or Not to Think: The Hidden Cost of Meta-Training with Excessive CoT Examples

**论文链接:** [http://arxiv.org/abs/2512.05318v1](http://arxiv.org/abs/2512.05318v1)

**作者:** Vignesh Kothapalli, Ata Fatahibaarzi, Hamed Firooz, Maziar Sanjabi

**发布时间:** 2025-12-04

**备注:** 26 pages, 45 figures, 3 tables

### GPT解析

### 总结

本研究针对预训练知识不足时，链式思维提示与少样本上下文学习结合在新任务上效果不佳的问题，提出了CoT-Recipe方法，通过调节CoT和非CoT示例的混合比例，显著提高了大型语言模型在新任务上的表现。

### 背景

链式思维提示与少样本上下文学习的结合已解锁大型语言模型的显著推理能力，但当预训练知识不足时，这种方法在新任务上表现不佳。

### 目的

研究在预训练知识有限的情况下，如何通过元训练技术使大型语言模型能够有效地学习新的抽象推理任务。

### 方法

使用CoT-ICL Lab框架在受控环境中研究问题，提出CoT-Recipe方法来调节元训练序列中CoT和非CoT示例的混合比例，以缓解CoT监督有限时性能下降的问题。

### 主要发现

虽然CoT示例有助于推理，但在元训练过程中过度使用会降低性能；通过CoT-Recipe仔细调节混合比例，即使在没有CoT示例的上下文中，也能将transformer在新任务上的准确率提高高达300%；应用于Qwen2.5系列模型进行符号推理任务，准确率提高高达130%。

### 结论

CoT-Recipe方法可以有效提高大型语言模型在缺乏预训练知识的新任务上的表现，具有广泛的适用性和显著的效果提升。

### 翻译

链式思维提示结合少样本上下文学习已解锁大型语言模型的显著推理能力。然而，当预训练知识不足时，使用CoT示例的上下文学习在新任务上效果不佳。我们在CoT-ICL Lab框架的受控环境中研究此问题，并提出元训练技术以在上下文中学习新的抽象推理任务。尽管CoT示例促进推理，但我们注意到在元训练过程中过度包含它们会降低性能，特别是在CoT监督有限的情况下。为缓解这种行为，我们提出了CoT-Recipe，这是一种调节元训练序列中CoT和非CoT示例混合比例的正式方法。我们证明，通过CoT-Recipe的仔细调节，即使在没有CoT示例的上下文中，也可以将transformer在新任务上的准确率提高高达300%。我们通过将这些技术应用于预训练的大型语言模型（Qwen2.5系列）进行符号推理任务，并观察到准确率提高高达130%，证实了这些技术的更广泛有效性。


### 论文摘要

Chain-of-thought (CoT) prompting combined with few-shot in-context learning (ICL) has unlocked significant reasoning capabilities in large language models (LLMs). However, ICL with CoT examples is ineffective on novel tasks when the pre-training knowledge is insufficient. We study this problem in a controlled setting using the CoT-ICL Lab framework, and propose meta-training techniques to learn novel abstract reasoning tasks in-context. Although CoT examples facilitate reasoning, we noticed that their excessive inclusion during meta-training degrades performance when CoT supervision is limited. To mitigate such behavior, we propose CoT-Recipe, a formal approach to modulate the mix of CoT and non-CoT examples in meta-training sequences. We demonstrate that careful modulation via CoT-Recipe can increase the accuracy of transformers on novel tasks by up to 300% even when there are no CoT examples available in-context. We confirm the broader effectiveness of these techniques by applying them to pretrained LLMs (Qwen2.5 series) for symbolic reasoning tasks and observing gains of up to 130% in accuracy.

---

## 102. STAR-GO: Improving Protein Function Prediction by Learning to Hierarchically Integrate Ontology-Informed Semantic Embeddings

**论文链接:** [http://arxiv.org/abs/2512.05245v1](http://arxiv.org/abs/2512.05245v1)

**作者:** Mehmet Efe Akça, Gökçe Uludoğan, Arzucan Özgür, İnci M. Baytaş

**发布时间:** 2025-12-04

**备注:** 14 pages, 2 figures, 6 tables

### GPT解析

### 总结

STAR-GO是一种基于Transformer的框架，通过联合建模GO术语的语义和结构特征来增强零样本蛋白质功能预测，实现了最先进的性能和优越的零样本泛化能力。

### 背景

蛋白质功能预测对于理解分子机制和促进生物及治疗发现至关重要，但实验注释速度远跟不上蛋白质序列数据的快速增长。

### 目的

开发一种能够有效处理新出现的GO术语的蛋白质功能预测方法，提高模型的泛化能力和适应性。

### 方法

STAR-GO整合了GO术语的文本定义和本体图结构来学习统一的GO表示，按层次顺序处理这些表示以传播信息，然后将这些表示与蛋白质序列嵌入对齐以捕获序列-功能关系。

### 主要发现

STAR-GO达到了最先进的性能和优越的零样本泛化能力，证明了整合语义和结构对于稳健和适应性强的蛋白质功能预测的效用。

### 结论

通过联合建模GO术语的语义和结构特征，STAR-GO为蛋白质功能预测提供了一种有效且适应性强的解决方案，能够更好地处理新出现的GO术语。

### 翻译

准确的蛋白质功能预测对于阐明分子机制和推进生物及治疗发现至关重要。然而，实验注释远远落后于蛋白质序列数据的快速增长。计算方法通过将蛋白质与基因本体论(GO)术语相关联来弥合这一差距，GO术语通过层次关系和文本定义编码功能知识。然而，现有模型往往强调一种模态而非另一种，限制了它们的泛化能力，特别是对于本体进化过程中频繁出现的新引入或未见的GO术语，这使得先前训练的模型过时。我们提出了STAR-GO，一种基于Transformer的框架，它联合建模GO术语的语义和结构特征，以增强零样本蛋白质功能预测。STAR-GO整合文本定义和本体图结构来学习统一的GO表示，这些表示按层次顺序处理，以从一般术语到特定术语传播信息。然后，这些表示与蛋白质序列嵌入对齐，以捕获序列-功能关系。STAR-GO实现了最先进的性能和优越的零样本泛化能力，证明了整合语义和结构对于稳健且适应性强的蛋白质功能预测的效用。代码可在https://github.com/boun-tabi-lifelu/stargo获取。


### 论文摘要

Accurate prediction of protein function is essential for elucidating molecular mechanisms and advancing biological and therapeutic discovery. Yet experimental annotation lags far behind the rapid growth of protein sequence data. Computational approaches address this gap by associating proteins with Gene Ontology (GO) terms, which encode functional knowledge through hierarchical relations and textual definitions. However, existing models often emphasize one modality over the other, limiting their ability to generalize, particularly to unseen or newly introduced GO terms that frequently arise as the ontology evolves, and making the previously trained models outdated. We present STAR-GO, a Transformer-based framework that jointly models the semantic and structural characteristics of GO terms to enhance zero-shot protein function prediction. STAR-GO integrates textual definitions with ontology graph structure to learn unified GO representations, which are processed in hierarchical order to propagate information from general to specific terms. These representations are then aligned with protein sequence embeddings to capture sequence-function relationships. STAR-GO achieves state-of-the-art performance and superior zero-shot generalization, demonstrating the utility of integrating semantics and structure for robust and adaptable protein function prediction. Code is available at https://github.com/boun-tabi-lifelu/stargo.

---

## 103. Designing an Optimal Sensor Network via Minimizing Information Loss

**论文链接:** [http://arxiv.org/abs/2512.05940v1](http://arxiv.org/abs/2512.05940v1)

**作者:** Daniel Waxman, Fernando Llorente, Katia Lamer, Petar M. Djurić

**发布时间:** 2025-12-05

**备注:** 37 pages, 15 figures. Accepted to Bayesian Analysis

### GPT解析

### 总结

该研究提出了一种新颖的基于物理模拟的传感器最优放置方法，用于监测时空过程。通过结合物理模拟与贝叶斯实验设计原理，开发出能从模拟数据中'最小化信息损失'的传感器网络优化算法，并在亚利桑那州凤凰城的气温监测案例中验证了其有效性。

### 背景

最优实验设计是统计学中的经典主题，但计算科学最新产生的大规模物理模拟数据很少被用于实验设计。本研究针对传感器放置问题，特别考虑时间维度在建模和优化中的作用。

### 目的

开发一种新的传感器放置方法，明确考虑时间维度，并利用基于物理模拟的大数据集来改进实验设计，从而最小化信息损失。

### 方法

引入基于模型的传感器放置标准和高效优化算法，结合物理模拟与贝叶斯实验设计原理，利用稀疏变分推断和可分离高斯-马尔可夫先验技术进行优化。

### 主要发现

在亚利桑那州凤凰城的气温监测案例研究中，该方法优于随机或准随机采样，特别是在传感器数量有限的情况下表现更佳。

### 结论

该框架具有实际应用价值，可扩展到更复杂的建模工具和现实世界部署场景，代表了实验设计领域的重要进步。

### 翻译

最优实验设计是统计学中的一个经典主题，有许多经过深入研究的问题、应用和解决方案。我们研究的设计问题是传感器的放置，用于监测时空过程，在我们的建模和优化中明确考虑时间维度。我们注意到，计算科学的最新进展通常产生基于物理模拟的大数据集，但这些数据集很少被用于实验设计。我们引入了一种新颖的基于模型的传感器放置标准，以及一种高效的优化算法，该算法结合了物理模拟和贝叶斯实验设计原理，以识别能从模拟数据中'最小化信息损失'的传感器网络。我们的技术依赖于稀疏变分推断和(可分离)高斯-马尔可夫先验，因此可以采用贝叶斯实验设计的多种技术。我们通过在亚利桑那州凤凰城使用最先进的物理模拟监测气温的案例研究验证了我们的方法。我们的结果表明，我们的框架优于随机或准随机采样，特别是在传感器数量有限的情况下。我们最后讨论了我们的框架的实际考虑和影响，包括更复杂的建模工具和现实世界的部署。


### 论文摘要

Optimal experimental design is a classic topic in statistics, with many well-studied problems, applications, and solutions. The design problem we study is the placement of sensors to monitor spatiotemporal processes, explicitly accounting for the temporal dimension in our modeling and optimization. We observe that recent advancements in computational sciences often yield large datasets based on physics-based simulations, which are rarely leveraged in experimental design. We introduce a novel model-based sensor placement criterion, along with a highly-efficient optimization algorithm, which integrates physics-based simulations and Bayesian experimental design principles to identify sensor networks that "minimize information loss" from simulated data. Our technique relies on sparse variational inference and (separable) Gauss-Markov priors, and thus may adapt many techniques from Bayesian experimental design. We validate our method through a case study monitoring air temperature in Phoenix, Arizona, using state-of-the-art physics-based simulations. Our results show our framework to be superior to random or quasi-random sampling, particularly with a limited number of sensors. We conclude by discussing practical considerations and implications of our framework, including more complex modeling tools and real-world deployments.

---

## 104. NeuroMemFPP: A recurrent neural approach for memory-aware parameter estimation in fractional Poisson process

**论文链接:** [http://arxiv.org/abs/2512.05893v1](http://arxiv.org/abs/2512.05893v1)

**作者:** Neha Gupta, Aditya Maheshwari

**发布时间:** 2025-12-05

**备注:** 12 pages

### GPT解析

### 总结

本文提出了一种基于循环神经网络的框架，用于估计分数泊松过程的参数，实验表明该方法比传统方法降低约55.3%的均方误差，并在真实世界数据上表现出色。

### 背景

分数泊松过程是一种能够建模具有记忆性和长程依赖性的事件到达过程的模型，但传统估计方法存在局限性。

### 目的

开发一种能够有效估计分数泊松过程关键参数μ和β的新方法，这些参数来自到达时间间隔序列。

### 方法

使用长短期记忆(LSTM)网络从到达时间间隔序列中估计参数，以捕捉时间依赖性特征。

### 主要发现

1)在合成数据上，与传统矩量法相比，均方误差降低约55.3%；2)方法在不同训练条件下表现可靠；3)在紧急呼叫记录和AAPL股票交易数据上有效跟踪日模式和参数变化。

### 结论

LSTM框架在处理具有复杂时间依赖性的真实世界数据时表现有效，为分数泊松过程的参数估计提供了新思路。

### 翻译

在本文中，我们提出了一种基于循环神经网络(RNN)的框架，用于估计分数泊松过程(FPP)的参数，该模型能够对具有记忆性和长程依赖性的事件到达进行建模。长短期记忆(LSTM)网络从到达时间间隔序列中估计关键参数μ>0和β∈(0,1)，有效地建模了它们的时间依赖性。我们在合成数据上的实验表明，与传统矩量法(MOM)相比，所提出的方法将均方误差(MSE)降低了约55.3%，并在不同的训练条件下表现可靠。我们在两个真实世界的高频数据集上测试了该方法：宾夕法尼亚州蒙哥马利县的紧急呼叫记录和AAPL股票交易数据。结果表明，LSTM可以有效跟踪日模式和参数变化，表明其在具有复杂时间依赖性的真实世界数据上的有效性。


### 论文摘要

In this paper, we propose a recurrent neural network (RNN)-based framework for estimating the parameters of the fractional Poisson process (FPP), which models event arrivals with memory and long-range dependence. The Long Short-Term Memory (LSTM) network estimates the key parameters $μ>0$ and $β\in(0,1)$ from sequences of inter-arrival times, effectively modeling their temporal dependencies. Our experiments on synthetic data show that the proposed approach reduces the mean squared error (MSE) by about 55.3\% compared to the traditional method of moments (MOM) and performs reliably across different training conditions. We tested the method on two real-world high-frequency datasets: emergency call records from Montgomery County, PA, and AAPL stock trading data. The results show that the LSTM can effectively track daily patterns and parameter changes, indicating its effectiveness on real-world data with complex time dependencies.

---

## 105. Fluctuating Environments Favor Extreme Dormancy Strategies and Penalize Intermediate Ones

**论文链接:** [http://arxiv.org/abs/2512.05856v1](http://arxiv.org/abs/2512.05856v1)

**作者:** Jorge Hidalgo, Lorenzo Fant, Rafael Rubio de Casas, Miguel A. Muñoz

**发布时间:** 2025-12-05

**备注:** 12 pages, 11 figures

### GPT解析

### 总结

本研究探讨了休眠策略如何与环境时间结构相互作用，特别是休眠持续时间如何影响种群适应性。通过模型和模拟，研究发现休眠持续时间与适应性之间存在非单调关系，存在三个不同的表现区域，且中间策略可能被自然选择所排斥。

### 背景

休眠是一种广泛存在的适应策略，使种群能够在波动环境中生存，但其益处如何依赖于环境变异性的时间结构尚不清楚。理解休眠与环境时间尺度的关系对于预测种群动态和进化具有重要意义。

### 目的

研究目的是探讨休眠策略如何与环境相关时间尺度相互作用，以及休眠持续时间如何影响种群适应性和进化策略。

### 方法

研究使用延迟逻辑模型，其中休眠个体在固定延迟后重新激活，出生率在时间相关的随机性下波动。通过数值模拟和分析计算，研究休眠与环境相关时间的相互作用。

### 主要发现

1. 休眠持续时间与适应性之间存在非单调关系，形成三个不同的表现区域
2. 极短休眠最大化线性增长但增加波动和灭绝风险
3. 极长休眠缓冲环境变异性，显著增加平均灭绝时间 despite 增长较慢
4. 中等休眠持续时间范围是适应不良的，同时降低增长和持久性
5. 进化基于模型证实了短休眠和长休眠策略之间的双稳态，避免中间延迟时间

### 结论

休眠持续时间不仅是生活史参数，更是适应环境时间尺度的机制。中间'危险的中间'策略可能被固有地排斥。这项工作确定了一个通用机制，即人口统计延迟与相关环境变异性相互作用产生非单调适应性景观，从而选择极端时间策略。

### 翻译

休眠是一种广泛存在的适应策略，使种群能够在波动环境中生存，但其益处如何依赖于环境变异性的时间结构尚不清楚。我们使用延迟逻辑模型研究休眠如何与环境相关时间相互作用，在该模型中，休眠个体在固定延迟后重新激活，而出生率在时间相关的随机性下波动。数值模拟和分析计算表明，人口统计记忆和有色乘性噪声的结合导致适应性对休眠持续时间有强烈的非单调依赖，形成三个不同的表现区域。极短休眠最大化线性增长但放大波动和灭绝风险。极长休眠缓冲环境变异性，尽管增长较慢，但显著增加平均灭绝时间。引人注目的是，我们发现一个广泛的中等休眠持续时间范围是适应不良的，由于延迟时间与环境自相关之间的不匹配，同时降低增长和持久性。一个基于进化体的模型证实了短休眠和长休眠策略之间的双稳态，这些策略避免中间延迟时间并进化为稳定的极端。


### 论文摘要

Dormancy is a widespread adaptive strategy that enables populations to persist in fluctuating environments, yet how its benefits depend on the temporal structure of environmental variability remains unclear. We examine how dormancy interacts with environmental correlation times using a delayed-logistic model in which dormant individuals reactivate after a fixed lag while birth rates fluctuate under temporally correlated stochasticity. Numerical simulations and analytical calculations show that the combination of demographic memory and colored multiplicative noise generates a strongly non-monotonic dependence of fitness on dormancy duration, with three distinct performance regimes. Very short dormancy maximizes linear growth but amplifies fluctuations and extinction risk. Very long dormancy buffers environmental variability, greatly increasing mean extinction times despite slower growth. Strikingly, we find a broad band of intermediate dormancy durations that is maladaptive, simultaneously reducing both growth and persistence due to a mismatch between delay times and environmental autocorrelation. An evolutionary agent-based model confirms bistability between short- and long-dormancy strategies, which avoid intermediate lag times and evolve toward stable extremes. These results show that dormancy duration is not merely a life-history parameter but an adaptive mechanism tuned to environmental timescales, and that intermediate "dangerous middle" strategies can be inherently disfavored. More broadly, this work identifies a generic mechanism by which demographic delays interacting with correlated environmental variability produce a non-monotonic fitness landscape that selects for extreme timing strategies.

---

## 106. USV: Unified Sparsification for Accelerating Video Diffusion Models

**论文链接:** [http://arxiv.org/abs/2512.05754v1](http://arxiv.org/abs/2512.05754v1)

**作者:** Xinjian Wu, Hongmei Wang, Yuan Zhou, Qinglin Lu

**发布时间:** 2025-12-05

### GPT解析

### 总结

USV是一种端到端可训练的框架，通过联合协调模型内部计算和采样过程中的稀疏化，解决了高保真视频扩散模型的可扩展性问题，实现了显著的加速效果。

### 背景

高保真视频扩散模型的可扩展性受到两个关键冗余源的制约：全局时空注意力的二次复杂性和长迭代去噪轨迹的计算开销。

### 目的

克服现有加速器只针对单一维度导致的收益递减问题，实现多维协同加速。

### 方法

提出USV（视频扩散模型的统一稀疏化），学习动态的、数据和时间步相关的稀疏化策略，修剪冗余注意力连接，合并语义相似令牌，减少去噪步骤，并将这些作为协调行动优化。

### 主要发现

多维协同设计使分离的加速策略相互强化；在去噪过程中实现高达83.3%的加速，端到端加速达到22.7%，同时保持高视觉保真度。

### 结论

统一、动态稀疏化是实现高效、高质量视频生成的实用路径。

### 翻译

高保真视频扩散模型的可扩展性受到两个关键冗余源的制约：全局时空注意力的二次复杂性和长迭代去噪轨迹的计算开销。现有的加速器--如稀疏注意力和步蒸馏采样器--通常单独针对单一维度，并很快遇到收益递减的问题，因为剩余的瓶颈成为主导。在这项工作中，我们引入了USV（视频扩散模型的统一稀疏化），这是一个端到端可训练的框架，通过联合协调模型内部计算和采样过程中的稀疏化来克服这一限制。USV学习动态的、数据和时间步相关的稀疏化策略，修剪冗余的注意力连接，自适应地合并语义相似的令牌，并减少去噪步骤，将它们不是作为独立的技巧，而是作为单个优化目标内的协调行动。这种多维协同设计使之前分离的加速策略之间能够产生强烈的相互强化。在大型视频生成基准上的广泛实验表明，USV在去噪过程中实现了高达83.3%的加速，端到端加速达到22.7%，同时保持高视觉保真度。我们的结果强调统一、动态稀疏化是实现高效、高质量视频生成的实用路径。


### 论文摘要

The scalability of high-fidelity video diffusion models (VDMs) is constrained by two key sources of redundancy: the quadratic complexity of global spatio-temporal attention and the computational overhead of long iterative denoising trajectories. Existing accelerators -- such as sparse attention and step-distilled samplers -- typically target a single dimension in isolation and quickly encounter diminishing returns, as the remaining bottlenecks become dominant. In this work, we introduce USV (Unified Sparsification for Video diffusion models), an end-to-end trainable framework that overcomes this limitation by jointly orchestrating sparsification across both the model's internal computation and its sampling process. USV learns a dynamic, data- and timestep-dependent sparsification policy that prunes redundant attention connections, adaptively merges semantically similar tokens, and reduces denoising steps, treating them not as independent tricks but as coordinated actions within a single optimization objective. This multi-dimensional co-design enables strong mutual reinforcement among previously disjoint acceleration strategies. Extensive experiments on large-scale video generation benchmarks demonstrate that USV achieves up to 83.3% speedup in the denoising process and 22.7% end-to-end acceleration, while maintaining high visual fidelity. Our results highlight unified, dynamic sparsification as a practical path toward efficient, high-quality video generation.

---

## 107. Emergence of Language in the Developing Brain

**论文链接:** [http://arxiv.org/abs/2512.05718v1](http://arxiv.org/abs/2512.05718v1)

**作者:** Linnea Evanson, Christine Bulteau, Mathilde Chipaux, Georg Dorfmüller, Sarah Ferrand-Sorbets, Emmanuel Raffo, Sarah Rosenberg, Pierre Bourdillon, Jean-Rémi King

**发布时间:** 2025-12-05

**备注:** *Equal contribution

### GPT解析

### 总结

研究通过记录儿童、青少年和成年人在听故事时的大脑活动，探索语言习得的神经机制，发现语言表征随年龄发展，大型语言模型能模拟这种神经发育轨迹。

### 背景

儿童仅需几百万个单词就能习得语言，但大脑中支持这种独特能力的机制仍不清楚。

### 目的

探究大脑中语言层次结构的定位、动态和发展，揭示语言习得的神经基础。

### 方法

记录46名儿童、青少年和成年人听《小王子》有声读物时的大脑活动，使用7400多个电极进行癫痫监测，并训练基于语言学理论或大型语言模型的神经编码和解码模型。

### 主要发现

各种语言特征在整个皮层都有稳健表征，甚至在2-5岁幼儿中也是如此；快速语音特征在幼儿颞上回已存在，而词汇级表征仅在较大年龄个体的联合皮层中出现；大型语言模型能自发捕捉这种神经发育轨迹。

### 结论

发育中大脑的语言表征随年龄成熟，现代AI系统为建模语言习得的神经基础提供了有希望的工具。

### 翻译

几百万个单词就足以让儿童习得语言。然而，支撑这一独特能力的大脑机制仍知之甚少。为解决这一问题，我们研究了46名儿童、青少年和成人在听《小王子》有声读物时的大脑活动，这些活动通过植入大脑的7400多个电极记录用于癫痫监测。然后，我们使用基于语言学理论或大型语言模型表征的神经编码和解码模型，来映射大脑中语言层次结构的定位、动态和发展。我们发现，各种语言特征在整个皮层都有稳健表征，即使在2-5岁的幼儿中也是如此。关键的是，这些表征随年龄发展：虽然快速语音特征在最小年龄个体的颞上回中已经存在，但较慢的词汇级表征仅在较大年龄个体的联合皮层中出现。值得注意的是，这种神经发育轨迹被大型语言模型自发捕捉：通过训练，这些AI模型学到的表征只能在成人脑中识别。这些发现共同揭示了发育中大脑语言表征的成熟过程，并表明现代AI系统为建模语言习得的神经基础提供了有希望的工具。


### 论文摘要

A few million words suffice for children to acquire language. Yet, the brain mechanisms underlying this unique ability remain poorly understood. To address this issue, we investigate neural activity recorded from over 7,400 electrodes implanted in the brains of 46 children, teenagers, and adults for epilepsy monitoring, as they listened to an audiobook version of "The Little Prince". We then train neural encoding and decoding models using representations, derived either from linguistic theory or from large language models, to map the location, dynamics and development of the language hierarchy in the brain. We find that a broad range of linguistic features is robustly represented across the cortex, even in 2-5-year-olds. Crucially, these representations evolve with age: while fast phonetic features are already present in the superior temporal gyrus of the youngest individuals, slower word-level representations only emerge in the associative cortices of older individuals. Remarkably, this neuro-developmental trajectory is spontaneously captured by large language models: with training, these AI models learned representations that can only be identified in the adult human brain. Together, these findings reveal the maturation of language representations in the developing brain and show that modern AI systems provide a promising tool to model the neural bases of language acquisition.

---

## 108. Randomness quantification in spontaneous emission

**论文链接:** [http://arxiv.org/abs/2512.05713v1](http://arxiv.org/abs/2512.05713v1)

**作者:** Chenxu Li, Shengfan Liu, Xiongfeng Ma

**发布时间:** 2025-12-05

**备注:** 13 pages, 2 figures

### GPT解析

### 总结

本研究开发了一个全面的量子信息理论框架，用于分析自发发射过程中的随机性生成，并比较了不同QRNG方案的安全性。

### 背景

量子相干性是生成内在随机性的基本资源，但基于自发发射的量子随机数生成器的随机性量化仍主要是现象学的，缺乏严格的敌手模型和量子相干性作用的清晰表征。

### 目的

开发一个全面的量子信息理论框架，用于自发发射过程中的随机性生成，并表征两种不同的窃听策略。

### 方法

分析两种敌手策略：直接访问原子系综和只访问其纯化；研究基于单光子检测和时间模式测量的QRNG；研究基于空间模式检测和相位波动的QRNG。

### 主要发现

单光子检测和时间模式测量的QRNG易受直接访问原子系综的敌手攻击，但仍能保证对访问纯化的敌手的内在随机性下限；基于空间模式检测和相位波动的QRNG对两种敌手都表现出安全性。

### 结论

提供了基于自发发射的QRNG方案的内在随机性的定量计算，为QRNG设计提供了更安全的选择。

### 翻译

量子相干性是生成内在随机性的基本资源，然而基于自发发射的量子随机数生成器的随机性量化仍然主要是现象学的。现有的随机性分析缺乏严格的敌手模型和对量子相干性在这些系统中作用的清晰表征。在本工作中，我们为自发发射过程中的随机性生成开发了一个全面的量子信息理论框架。我们表征了两种不同的窃听策略：一种是敌手直接访问原子系综，另一种是敌手只访问其纯化。我们的分析表明，当随机性通过单光子检测和时间模式测量生成时，QRNG容易受到第一种敌手场景的攻击，尽管即使在原子最大信息泄露的情况下，它仍然能保证对第二种敌手场景的内在随机性下限。相比之下，基于空间模式检测和相位波动的QRNG对两种类型的敌手都表现出安全性，提供稳健的随机性生成。此外，我们提供了这些基于自发发射的QRNG方案的内在随机性的定量计算。


### 论文摘要

Quantum coherence serves as a fundamental resource for generating intrinsic randomness, yet the quantification of randomness in quantum random number generators (QRNGs) based on spontaneous emission has remained largely phenomenological. Existing randomness analysis lacks rigorous adversarial models and a clear characterization of the role of quantum coherence in these systems. In this work, we develop a comprehensive quantum information-theoretic framework for randomness generation in spontaneous emission processes. We characterize two distinct eavesdropping strategies: one where the adversary directly accesses the atom ensemble, and the other where the adversary accesses only its purification. Our analysis reveals that when randomness is generated through single-photon detection and temporal mode measurements, the QRNG is vulnerable to the first adversary scenario, though it still guarantees a lower bound on intrinsic randomness against the second adversary scenario even under maximal information leakage from the atoms. In contrast, QRNGs based on spatial mode detection and phase fluctuations demonstrate security against both types of adversaries, providing robust randomness generation. Furthermore, we provide a quantitative calculation of intrinsic randomness for these spontaneous-emission-based QRNG schemes.

---

## 109. Scenario-aware Uncertainty Quantification for Trajectory Prediction with Statistical Guarantees

**论文链接:** [http://arxiv.org/abs/2512.05682v1](http://arxiv.org/abs/2512.05682v1)

**作者:** Yiming Shu, Jiahui Xu, Linghuan Kong, Fangni Zhang, Guodong Yin, Chen Sun

**发布时间:** 2025-12-05

### GPT解析

### 总结

该研究提出了一种新颖的场景感知不确定性量化框架，用于自动驾驶系统中轨迹预测的可靠不确定性量化和可靠性评估，解决了现有深度学习预测器在异构现实场景中缺乏不确定性感知框架的问题。

### 背景

在安全关键型自动驾驶系统中，轨迹预测的可靠不确定性量化至关重要，然而现有的深度学习预测器缺乏可适应异构现实场景的不确定性感知框架。

### 目的

为了填补这一空白，研究提出了一种新颖的场景感知不确定性量化框架，为预测的轨迹提供预测区间和可靠性评估。

### 方法

1) 将预测轨迹和真实轨迹投影到Frenet坐标系中的地图参考路线上；2) 使用CopulaCPTS保形校准方法为不同场景生成时间预测区间；3) 在轨迹可靠性鉴别器中协同分析平均误差和校准置信区间建立可靠性模型；4) 利用联合风险模型识别关键点，将轨迹分割为可靠和不可靠段。

### 主要发现

使用真实nuPlan数据集的评估表明，该框架在不同驾驶场景中能有效进行场景感知不确定性量化和可靠性评估。

### 结论

该研究提出的框架能够为下游规划模块提供可操作的可靠性结果，提高了自动驾驶系统中轨迹预测的安全性和可靠性。

### 翻译

在安全关键型自动驾驶系统中，轨迹预测的可靠不确定性量化至关重要，然而现有的深度学习预测器缺乏可适应异构现实场景的不确定性感知框架。为了填补这一空白，我们提出了一种新颖的场景感知不确定性量化框架，为预测的轨迹提供预测区间和可靠性评估。首先，将训练好的预测器预测的轨迹及其真实值投影到Frenet坐标系中的地图导出的参考路线上。然后，我们使用CopulaCPTS作为保形校准方法，为不同场景生成时间预测区间作为不确定性度量。基于此，在提出的轨迹可靠性鉴别器中，协同分析平均误差和校准置信区间，为不同场景建立可靠性模型。随后，风险感知鉴别器利用结合Frenet坐标系中纵向和横向预测区间的联合风险模型来识别关键点。这使得轨迹能够被分割为可靠和不可靠的段，具有为下游规划模块提供可操作可靠性结果的优势。我们使用真实的nuPlan数据集评估了我们的框架，证明了其在不同驾驶场景中场景感知不确定性量化和可靠性评估的有效性。


### 论文摘要

Reliable uncertainty quantification in trajectory prediction is crucial for safety-critical autonomous driving systems, yet existing deep learning predictors lack uncertainty-aware frameworks adaptable to heterogeneous real-world scenarios. To bridge this gap, we propose a novel scenario-aware uncertainty quantification framework to provide the predicted trajectories with prediction intervals and reliability assessment. To begin with, predicted trajectories from the trained predictor and their ground truth are projected onto the map-derived reference routes within the Frenet coordinate system. We then employ CopulaCPTS as the conformal calibration method to generate temporal prediction intervals for distinct scenarios as the uncertainty measure. Building upon this, within the proposed trajectory reliability discriminator (TRD), mean error and calibrated confidence intervals are synergistically analyzed to establish reliability models for different scenarios. Subsequently, the risk-aware discriminator leverages a joint risk model that integrates longitudinal and lateral prediction intervals within the Frenet coordinate to identify critical points. This enables segmentation of trajectories into reliable and unreliable segments, holding the advantage of informing downstream planning modules with actionable reliability results. We evaluated our framework using the real-world nuPlan dataset, demonstrating its effectiveness in scenario-aware uncertainty quantification and reliability assessment across diverse driving contexts.

---

## 110. Deep Learning-Based Real-Time Sequential Facial Expression Analysis Using Geometric Features

**论文链接:** [http://arxiv.org/abs/2512.05669v1](http://arxiv.org/abs/2512.05669v1)

**作者:** Talha Enes Koksal, Abdurrahman Gumus

**发布时间:** 2025-12-05

### GPT解析

### 总结

本研究提出了一种基于深度学习和几何特征的实时序列面部表情识别新方法，使用MediaPipe FaceMesh进行面部标志点检测，结合时间动态分析和ConvLSTM1D网络进行分类，在多个数据集上实现了高准确率并具有实时处理能力。

### 背景

面部表情识别是增强人机交互和发展情感感知系统的重要组成部分，实时检测和解释面部表情对于用户体验个性化到智能监控系统等各种应用变得越来越重要。

### 目的

提出一种使用深度学习和几何特征进行实时序列面部表情识别的新方法。

### 方法

使用MediaPipe FaceMesh进行快速准确的面部标志点检测，提取包括欧几里得距离和角度的几何特征，通过分析连续帧之间的特征差异融入时间动态，使用ConvLSTM1D网络后接多层感知器块进行分类，并在多个公开数据集上评估性能。

### 主要发现

在CK+、Oulu-CASIA（VIS和NIR）和MMI数据集上分别达到了93%、79%、77%和68%的准确率，模型具有良好的泛化能力，在消费级硬件上每秒可处理约165帧，具有实时适用性。

### 结论

该研究通过提供快速、准确且适应性强的解决方案对面部表情分析领域做出贡献，强调了情感感知技术和个性化用户体验进一步发展的潜力，源代码已在GitHub上公开以促进该领域进一步研究。

### 翻译

面部表情识别是增强人机交互和发展情感感知系统的重要组成部分。实时检测和解释面部表情对于各种应用变得越来越重要，从用户体验个性化到智能监控系统。本研究提出了一种使用深度学习和几何特征进行实时序列面部表情识别的新方法。该方法利用MediaPipe FaceMesh进行快速准确的面部标志点检测。从这些标志点提取几何特征，包括欧几里得距离和角度。通过分析连续帧之间的特征差异来融入时间动态，从而能够检测表情的开始、顶点和结束阶段。对于分类，使用ConvLSTM1D网络后接多层感知器块。该方法在多个公开可用数据集上进行了评估，包括CK+、Oulu-CASIA（VIS和NIR）和MMI。分别达到了93%、79%、77%和68%的准确率。还进行了复合数据集实验以评估模型的泛化能力。该方法展示了实时适用性，在消费级硬件上每秒处理约165帧。这项研究通过提供快速、准确且适应性强的解决方案，对面部表情分析领域做出了贡献。研究结果强调了情感感知技术和个性化用户体验进一步发展的潜力，为更复杂的人机交互系统铺平了道路。为了促进该领域的进一步研究，本研究的完整源代码已在GitHub上公开：https://github.com/miralab-ai/facial-expression-analysis。


### 论文摘要

Facial expression recognition is a crucial component in enhancing human-computer interaction and developing emotion-aware systems. Real-time detection and interpretation of facial expressions have become increasingly important for various applications, from user experience personalization to intelligent surveillance systems. This study presents a novel approach to real-time sequential facial expression recognition using deep learning and geometric features. The proposed method utilizes MediaPipe FaceMesh for rapid and accurate facial landmark detection. Geometric features, including Euclidean distances and angles, are extracted from these landmarks. Temporal dynamics are incorporated by analyzing feature differences between consecutive frames, enabling the detection of onset, apex, and offset phases of expressions. For classification, a ConvLSTM1D network followed by multilayer perceptron blocks is employed. The method's performance was evaluated on multiple publicly available datasets, including CK+, Oulu-CASIA (VIS and NIR), and MMI. Accuracies of 93%, 79%, 77%, and 68% were achieved respectively. Experiments with composite datasets were also conducted to assess the model's generalization capabilities. The approach demonstrated real-time applicability, processing approximately 165 frames per second on consumer-grade hardware. This research contributes to the field of facial expression analysis by providing a fast, accurate, and adaptable solution. The findings highlight the potential for further advancements in emotion-aware technologies and personalized user experiences, paving the way for more sophisticated human-computer interaction systems. To facilitate further research in this field, the complete source code for this study has been made publicly available on GitHub: https://github.com/miralab-ai/facial-expression-analysis.

---

## 111. Executing Discrete/Continuous Declarative Process Specifications via Complex Event Processing

**论文链接:** [http://arxiv.org/abs/2512.05653v1](http://arxiv.org/abs/2512.05653v1)

**作者:** Stefan Schönig, Leo Poss, Fabrizio Maria Maggi

**发布时间:** 2025-12-05

**备注:** Preprint

### GPT解析

### 总结

本文提出了一种基于复杂事件处理(CEP)的执行架构，实现了混合声明式模型的实时执行和强制执行，弥合了混合规范与操作控制之间的差距。

### 背景

传统业务流程管理(BPM)专注于离散事件，未能融合网络物理环境中的关键连续传感器数据。虽然混合声明式规范使用信号时序逻辑(STL)解决了这一问题，允许同时约束离散事件和实值信号，但现有工作仅限于监控和事后一致性检查。

### 目的

开发一种能够实时执行和强制执行混合声明式模型的执行架构，使系统能够根据连续传感器行为主动触发活动和强制执行流程边界。

### 方法

采用三层方法，将STL启发的谓词集成到执行流程中，构建基于复杂事件处理(CEP)的执行架构。

### 主要发现

通过将STL启发的谓词集成到执行流程中，系统能够根据连续传感器行为主动触发活动并强制执行流程边界，实现了从混合规范到操作控制的桥梁。

### 结论

该执行架构弥合了混合规范与操作控制之间的差距，使混合声明式模型能够在实际业务流程中实时执行和强制执行。

### 翻译

传统业务流程管理(BPM)专注于离散事件，未能融合网络物理环境中的关键连续传感器数据。利用信号时序逻辑(STL)的混合声明式规范通过允许同时约束离散事件和实值信号解决了这一局限性。然而，现有工作仅限于监控和事后一致性检查。本文引入了一种新颖的基于复杂事件处理(CEP)的执行架构，能够实时执行和强制执行混合声明式模型。我们的三层方法将STL启发的谓词集成到执行流程中，使系统能够根据连续传感器行为主动触发活动并强制执行流程边界。这种方法弥合了混合规范与操作控制之间的差距。


### 论文摘要

Traditional Business Process Management (BPM) focuses on discrete events and fails to incorporate critical continuous sensor data in cyber-physical environments. Hybrid declarative specifications, utilizing Signal Temporal Logic (STL), address this limitation by allowing constraints over both discrete events and real-valued signals. However, existing work has been limited to monitoring and post-hoc conformance checking. This paper introduces a novel Complex Event Processing (CEP)-based execution architecture that enables the real-time execution and enforcement of hybrid declarative models. Our three-layer approach integrates STL-inspired predicates into the execution flow, allowing the system to actively trigger activities and enforce process boundaries based on continuous sensor behavior. This approach bridges the gap between hybrid specification and operational control.

---

## 112. The Topology of Hardship: Empirical Curriculum Graphs and Structural Bottlenecks in Engineering Degrees

**论文链接:** [http://arxiv.org/abs/2512.05561v1](http://arxiv.org/abs/2512.05561v1)

**作者:** H. R. Paz

**发布时间:** 2025-12-05

**备注:** 24 pages, 4 figures, 3 tables

### GPT解析

### 总结

该研究引入了'困难拓扑'概念，通过分析学生实际学习轨迹，将课程难度量化为可测量的拓扑特性，发现结构密集、瓶颈多的课程与高辍学率和学习时间延长显著相关。

### 背景

工程学位通常被认为是'困难的'，但这种困难通常被讨论为内容难度或学生弱点，而非课程结构特性。现有研究多依赖官方教学大纲而非学生实际学习路径。

### 目的

引入'困难拓扑'概念，从学生实际轨迹中定量描述长期工程项目的课程复杂性，并研究其与辍学率和学位完成时间的关系。

### 方法

基于CAPIRE多级轨迹建模框架，从多个队列的入学和完成数据重建29个工程课程图，计算结构指标和实证困难指标，组合成综合困难指数并与辍学率和学位获得时间关联。

### 主要发现

课程难度不是模糊认知，而是可测量的拓扑特性：少数结构密集、瓶颈多的课程占不成比例的辍学和临时不同步比例。

### 结论

课程拓扑结构对学习成效有显著影响，为课程改革、认证和数据驱动的政策设计提供了重要启示。

### 翻译

工程学位通常被认为是'困难的'，但这种困难通常被讨论为内容难度或学生弱点，而不是课程本身的结构特性。近期关于课程先修网络和课程图的研究表明，学习计划可以建模为具有可识别枢纽和瓶颈的复杂网络，但大多数研究依赖于官方教学大纲，而非学生实际如何通过系统学习的情况（Simon de Blas等人，2021；Stavrinides & Zuev，2023；Yang等人，2024；Wang等人，2025）。本文引入了困难拓扑的概念：一种从长期工程项目的学生实际轨迹中推导出的课程复杂性定量描述。基于CAPIRE多级轨迹建模框架（Paz，2025a，2025b），我们从多个队列的入学和完成数据重建了29个工程课程图。对每个图计算结构指标（如密度、最长路径、瓶颈中心性）和实证困难指标，包括阻塞概率和进展时间。将这些指标组合成一个综合困难指数，然后与观察到的辍学率和学位获得时间相关联。我们的研究结果表明，课程难度不是一种模糊的认知，而是一种可测量的拓扑特性：少数结构密集、瓶颈多的课程占不成比例的辍学和临时不同步比例。我们讨论了对课程改革、认证和数据驱动的政策设计的启示。


### 论文摘要

Engineering degrees are often perceived as "hard", yet this hardness is usually discussed in terms of content difficulty or student weaknesses rather than as a structural property of the curriculum itself. Recent work on course-prerequisite networks and curriculum graphs has shown that study plans can be modelled as complex networks with identifiable hubs and bottlenecks, but most studies rely on official syllabi rather than on how students actually progress through the system (Simon de Blas et al., 2021; Stavrinides & Zuev, 2023; Yang et al., 2024; Wang et al., 2025).   This paper introduces the notion of topology of hardship: a quantitative description of curriculum complexity derived from empirical student trajectories in long-cycle engineering programmes. Building on the CAPIRE framework for multilevel trajectory modelling (Paz, 2025a, 2025b), we reconstruct degree-curriculum graphs from enrolment and completion data for 29 engineering curricula across several cohorts. For each graph we compute structural metrics (e.g., density, longest path, bottleneck centrality) and empirical hardship measures capturing blocking probability and time-to-progress. These are combined into a composite hardship index, which is then related to observed dropout rates and time to degree.   Our findings show that curriculum hardness is not a vague perception but a measurable topological property: a small number of structurally dense, bottleneck-heavy curricula account for a disproportionate share of dropout and temporal desynchronisation. We discuss implications for curriculum reform, accreditation, and data-informed policy design.

---

## 113. VOST-SGG: VLM-Aided One-Stage Spatio-Temporal Scene Graph Generation

**论文链接:** [http://arxiv.org/abs/2512.05524v1](http://arxiv.org/abs/2512.05524v1)

**作者:** Chinthani Sugandhika, Chen Li, Deepu Rajan, Basura Fernando

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种名为VOST-SGG的VLM辅助单阶段时空场景图生成框架，通过引入双源查询初始化策略和多模态特征库，解决了现有ST-SGG模型中可学习查询语义不足和仅依赖单模态视觉特征的局限性，在Action Genome数据集上实现了最先进的性能。

### 背景

时空场景图生成（ST-SGG）旨在建模视频帧中的对象及其随时间演变的关系，为视频字幕生成和视觉问答等下游推理任务提供可解释的表示。尽管基于DETR风格的单阶段ST-SGG模型最近有所进展，但仍存在两个关键局限性：可学习查询在语义上信息不足且实例无关地初始化；模型仅依赖单模态视觉特征进行谓词分类。

### 目的

解决现有ST-SGG模型的关键局限性，通过整合视觉-语言模型（VLM）的常识推理能力，提高时空场景图生成的性能。

### 方法

提出VOST-SGG框架，包含两个创新点：1) 双源查询初始化策略，将'关注什么'与'关注哪里'分离，实现语义基础的what-where推理；2) 多模态特征库，融合来自VLM的视觉、文本和空间线索，改进谓词分类。

### 主要发现

在Action Genome数据集上的大量实验表明，VOST-SGG方法达到了最先进的性能，验证了整合VLM辅助的语义先验和多模态特征对ST-SGG的有效性。

### 结论

将VLM辅助的语义先验和多模态特征整合到ST-SGG流程中能够显著提高模型性能，作者已将代码发布至https://github.com/LUNAProject22/VOST。

### 翻译

时空场景图生成（ST-SGG）旨在对视频帧中的对象及其随时间演变的关系进行建模，为视频字幕生成和视觉问答等下游推理任务提供可解释的表示。尽管最近基于DETR风格的单阶段ST-SGG模型有所进展，但它们仍存在几个关键局限性。首先，虽然这些模型依赖于基于注意力的可学习查询作为核心组件，但这些可学习查询在语义上信息不足且实例无关地初始化。其次，这些模型完全依赖单模态视觉特征进行谓词分类。为解决这些挑战，我们提出了VOST-SGG，这是一种VLM辅助的单阶段ST-SGG框架，将视觉-语言模型（VLM）的常识推理能力整合到ST-SGG流程中。首先，我们引入了双源查询初始化策略，将'关注什么'与'关注哪里'分离，实现语义基础的what-where推理。此外，我们提出了一个多模态特征库，融合来自VLM的视觉、文本和空间线索，以改进谓词分类。在Action Genome数据集上的大量实验表明，我们的方法达到了最先进的性能，验证了将VLM辅助的语义先验和多模态特征整合到ST-SGG中的有效性。我们将在https://github.com/LUNAProject22/VOST发布代码。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决时空场景图生成（ST-SGG）中的三个关键问题：1）现有模型的可学习查询缺乏语义特异性；2）查询初始化不考虑特定视频帧的上下文；3）仅依赖视觉特征进行谓词分类。这个问题很重要，因为ST-SGG能对视频中的对象及其随时间变化的关系进行建模，为视频字幕生成、视觉问答等任务提供可解释表示，对实现监控、机器人等领域的动态场景理解至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有单阶段ST-SGG模型的局限性，然后认识到视觉语言模型（VLMs）的常识推理能力可以弥补这些不足。他们借鉴了DETR风格的Transformer架构和OED框架作为基础，但进行了两个关键创新：双源查询初始化策略和多模态特征库。他们使用Qwen2.5-VL-7B-Instruct作为VLM，并参考了MS-COCO数据集进行预训练，将VLM的常识知识集成到ST-SGG流程中。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是利用视觉语言模型（VLM）的常识推理能力增强时空场景图生成，通过分离'关注什么'和'从哪里关注'，并融合多模态信息来改善关系识别。整体流程：1）输入视频帧并提取视觉特征；2）使用VLM生成合理的主体、客体和谓词候选；3）构建多模态特征库；4）初始化双源查询；5）通过主体-客体解码器和谓词解码器分别处理特征；6）进行时间聚合；7）生成最终的时空场景图。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1）双源查询初始化策略：分离语义内容查询和空间位置查询，提供实例特定语义指导和空间稳定性；2）多模态特征库：融合视觉、文本和空间线索，改善谓词分类。相比之前的工作，不同之处在于：使用VLM提供的语义信息而非零向量初始化查询；融合多模态而非仅依赖视觉特征；提供实例特定和实例无关的查询初始化分离；在识别罕见关系方面表现出更好的性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了VOST-SGG，一种通过双源查询初始化和多模态特征库集成视觉语言模型常识推理能力的新型时空场景图生成框架，显著提升了模型在识别对象关系特别是罕见关系方面的性能。'}


### 论文摘要

Spatio-temporal scene graph generation (ST-SGG) aims to model objects and their evolving relationships across video frames, enabling interpretable representations for downstream reasoning tasks such as video captioning and visual question answering. Despite recent advancements in DETR-style single-stage ST-SGG models, they still suffer from several key limitations. First, while these models rely on attention-based learnable queries as a core component, these learnable queries are semantically uninformed and instance-agnostically initialized. Second, these models rely exclusively on unimodal visual features for predicate classification. To address these challenges, we propose VOST-SGG, a VLM-aided one-stage ST-SGG framework that integrates the common sense reasoning capabilities of vision-language models (VLMs) into the ST-SGG pipeline. First, we introduce the dual-source query initialization strategy that disentangles what to attend to from where to attend, enabling semantically grounded what-where reasoning. Furthermore, we propose a multi-modal feature bank that fuses visual, textual, and spatial cues derived from VLMs for improved predicate classification. Extensive experiments on the Action Genome dataset demonstrate that our approach achieves state-of-the-art performance, validating the effectiveness of integrating VLM-aided semantic priors and multi-modal features for ST-SGG. We will release the code at https://github.com/LUNAProject22/VOST.

---

## 114. Multi-state Modeling of Delay Evolution in Suburban Rail Transports

**论文链接:** [http://arxiv.org/abs/2512.05521v1](http://arxiv.org/abs/2512.05521v1)

**作者:** Stefania Colombo, Alfredo Gimenez Zapiola, Francesca Ieva, Simone Vantini

**发布时间:** 2025-12-05

### GPT解析

### 总结

该研究应用连续时间多状态模型分析意大利伦巴第地区S5郊区铁路线的延误动态，考虑了运营、气象和背景数据，发现延误动态因旅行方向、时间段和路线段而异，车站饱和度和乘客负荷等因素显著影响延误升级或恢复风险。

### 背景

铁路延误是铁路系统中的持续性问题，尤其在郊区铁路网络中，由于频繁服务和大量乘客，运营复杂性增加。传统延误模型往往忽视真实延误传播的时间动态和结构动态。

### 目的

应用连续时间多状态模型分析伦巴第地区S5郊区铁路线上延误的时间演变，使用详细的运营、气象和背景数据建模延误转换，同时考虑可观测的异质性。

### 方法

采用连续时间多状态模型，分析意大利伦巴第地区的S5郊区铁路线，利用详细的运营数据、气象数据和背景数据，对延误转换进行建模，并考虑可观测的异质性。

### 主要发现

延误动态会根据旅行方向、时间段和路线段而变化；车站饱和度和乘客负荷等因素对延误升级或恢复的风险有显著影响。

### 结论

该研究在方法论上取得了进展，同时提供了提高铁路服务可靠性的实用结果。

### 翻译

铁路延误是铁路系统中持续存在的问题，特别是在郊区铁路网络中，由于频繁服务和大量乘客，运营复杂性增加。传统延误模型往往忽视真实延误传播的时间动态和结构动态。这项工作应用连续时间多状态模型分析意大利伦巴第地区S5郊区铁路线上延误的时间演变。利用详细的运营、气象和背景数据，该研究建模了延误转换，同时考虑了可观测的异质性。研究结果显示延误动态因旅行方向、时间段和路线段而异。诸如车站饱和度和乘客负荷等因素被证明显著影响延误升级或恢复的风险。该研究既提供了方法上的进展，也提供了提高铁路服务可靠性的实用结果。


### 论文摘要

Train delays are a persistent issue in railway systems, particularly in suburban networks where operational complexity is heightened by frequent services and high passenger volumes. Traditional delay models often overlook the temporal and structural dynamics of real delay propagation.   This work applies continuous-time multi-state models to analyze the temporal evolution of delay on the S5 suburban line in Lombardy, Italy. Using detailed operational, meteorological, and contextual data, the study models delay transitions while accounting for observable heterogeneity.   The findings reveal how delay dynamics vary by travel direction, time slot, and route segment. Covariates such as station saturation and passenger load are shown to significantly affect the risk of delay escalation or recovery. The study offers both methodological advancements and practical results for improving the reliability of rail services.

---

## 115. Spatiotemporal Tubes for Differential Drive Robots with Model Uncertainty

**论文链接:** [http://arxiv.org/abs/2512.05495v1](http://arxiv.org/abs/2512.05495v1)

**作者:** Ratnangshu Das, Ahan Basu, Christos Verginis, Pushpak Jagtap

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种基于时空管的控制框架，用于处理具有动态不确定性和外部干扰的差分驱动移动机器人，保证满足时间到达-避开-停留规范，并通过仿真验证了其优越性。

### 背景

差分驱动移动机器人在实际应用中面临动态不确定性和外部干扰的挑战，需要一种能够保证满足时间到达-避开-停留规范的鲁棒控制方法。

### 目的

开发一种基于时空管的控制框架，使差分驱动移动机器人能够在动态不确定性和外部干扰环境下，从起始区域安全到达目标区域并避开障碍物，同时满足时间规范要求。

### 方法

采用圆形时空管定义动态安全走廊，开发基于采样的综合算法构建可行STT，设计解析的、无近似的控制律确保机器人保持在管内，并通过仿真验证方法的有效性。

### 主要发现

所提出的控制器计算效率高，对干扰和模型不确定性具有鲁棒性，不需要模型近似或在线优化，在仿真中展现出优越的鲁棒性、准确性和计算效率。

### 结论

基于时空管的控制框架能够有效处理差分驱动移动机器人在动态不确定性和外部干扰环境下的导航问题，保证满足时间到达-避开-停留规范，是一种高效鲁棒的解决方案。

### 翻译

本文提出了一种基于时空管(STT)的控制框架，用于处理具有动态不确定性和外部干扰的差分驱动移动机器人，保证满足时间到达-避开-停留(T-RAS)规范。该方法采用圆形STT，特点是中心和平滑变化的时间半径，定义动态安全走廊，引导机器人从起始区域到目标区域同时避开障碍物。特别是，我们首先开发了一种基于采样的综合算法，构建满足规定时间和安全约束的可行STT，并有形式保证。为了确保机器人保持在管内，我们随后设计了一个解析的、无近似的控制律。由此产生的控制器计算效率高，对干扰和模型不确定性具有鲁棒性，且不需要模型近似或在线优化。所提出的框架通过差分驱动机器人的仿真研究进行了验证，并与最先进的方法进行了基准测试，展示了优越的鲁棒性、准确性和计算效率。


### 论文摘要

This paper presents a Spatiotemporal Tube (STT)-based control framework for differential-drive mobile robots with dynamic uncertainties and external disturbances, guaranteeing the satisfaction of Temporal Reach-Avoid-Stay (T-RAS) specifications. The approach employs circular STT, characterized by smoothly time-varying center and radius, to define dynamic safe corridors that guide the robot from the start region to the goal while avoiding obstacles. In particular, we first develop a sampling-based synthesis algorithm to construct a feasible STT that satisfies the prescribed timing and safety constraints with formal guarantees. To ensure that the robot remains confined within this tube, we then design analytically a closed-form, approximation-free control law. The resulting controller is computationally efficient, robust to disturbances and {model uncertainties}, and requires no model approximations or online optimization. The proposed framework is validated through simulation studies on a differential-drive robot and benchmarked against state-of-the-art methods, demonstrating superior robustness, accuracy, and computational efficiency.

---

## 116. WaterWave: Bridging Underwater Image Enhancement into Video Streams via Wavelet-based Temporal Consistency Field

**论文链接:** [http://arxiv.org/abs/2512.05492v1](http://arxiv.org/abs/2512.05492v1)

**作者:** Qi Zhu, Jingyi Zhang, Naishan Zheng, Wei Yu, Jinghao Zhang, Deyi Ji, Feng Zhao

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种名为WaterWave的水下视频增强方法，解决了现有方法缺乏时间一致性的问题。该方法基于小波时间一致性场，通过隐式表示方式增强视频信号，并设计了水下流校正模块以提高时间频带的表示准确性。

### 背景

水下视频对获取困难，因为水下成像复杂。现有视频水下增强方法通常是将单图像增强模型逐帧应用，导致缺乏时间一致性的问题。

### 目的

解决水下视频增强中的时间一致性问题，提出一种能在无配对数据条件下工作的视频增强方法。

### 方法

重新思考自然视频中的时间流形，从局部时间频率角度观察动态场景中的时间一致性先验。提出基于小波时间一致性场的隐式表示方式WaterWave，在先验约束下渐进式过滤和衰减不一致成分，同时保持运动细节和场景。设计水下流校正模块以更准确地表示时间频带，考虑水下场景中的传输。

### 主要发现

WaterWave显著提高了使用单图像水下增强生成的视频质量。在下游水下跟踪任务UOSTrack和MAT上表现优异，相比原始视频，精确度分别提高了19.7%和9.7%。

### 结论

WaterWave是一种有效的水下视频增强方法，解决了时间一致性问题，并在下游任务中展现出良好的应用潜力。

### 翻译

水下视频由于水下成像复杂而难以获取。在这种情况下，大多数现有的视频水下增强方法是通过逐帧直接应用单图像增强模型来执行的，但一个自然的问题是缺乏时间一致性。为了缓解这个问题，我们重新思考了自然视频中固有的时间流形，并从局部时间频率角度观察了动态场景中的时间一致性先验。基于特定的先验和无配对数据条件，我们提出了一种用于增强视频信号的隐式表示方式，该方式在小波时间一致性场中进行，称为WaterWave。具体来说，在先验的约束下，我们渐进式地过滤和衰减不一致的成分，同时保持运动细节和场景，实现自然流畅的视频。此外，为了更准确地表示时间频带，设计了一个水下流校正模块，以考虑水下场景中的传输来校正估计的流。大量实验表明，WaterWave显著提高了使用单图像水下增强生成的视频质量。此外，我们的方法在下游水下跟踪任务中显示出很高的潜力，如UOSTrack和MAT，以较大优势优于原始视频，即在精确度上分别提高了19.7%和9.7%。


### 论文摘要

Underwater video pairs are fairly difficult to obtain due to the complex underwater imaging. In this case, most existing video underwater enhancement methods are performed by directly applying the single-image enhancement model frame by frame, but a natural issue is lacking temporal consistency. To relieve the problem, we rethink the temporal manifold inherent in natural videos and observe a temporal consistency prior in dynamic scenes from the local temporal frequency perspective. Building upon the specific prior and no paired-data condition, we propose an implicit representation manner for enhanced video signals, which is conducted in the wavelet-based temporal consistency field, WaterWave. Specifically, under the constraints of the prior, we progressively filter and attenuate the inconsistent components while preserving motion details and scenes, achieving a natural-flowing video. Furthermore, to represent temporal frequency bands more accurately, an underwater flow correction module is designed to rectify estimated flows considering the transmission in underwater scenes. Extensive experiments demonstrate that WaterWave significantly enhances the quality of videos generated using single-image underwater enhancements. Additionally, our method demonstrates high potential in downstream underwater tracking tasks, such as UOSTrack and MAT, outperforming the original video by a large margin, i.e., 19.7% and 9.7% on precise respectively.

---

## 117. Turbulence Regression

**论文链接:** [http://arxiv.org/abs/2512.05483v1](http://arxiv.org/abs/2512.05483v1)

**作者:** Yingang Fan, Binjie Ding, Baiyi Chen

**发布时间:** 2025-12-05

### GPT解析

### 总结

该研究提出了一种NeuTucker分解模型，用于处理连续但稀疏的三维风场数据，以改进低空湍流的预测准确性。

### 背景

气流湍流是由速度、压力或方向的剧烈变化产生的无序和不规则运动状态。多种复杂因素导致复杂的低空湍流结果。在当前观测条件下，特别是仅使用风廓线雷达数据时，传统方法难以准确预测湍流状态。

### 目的

引入一种使用离散数据的NeuTucker分解模型，构建基于Tucker神经网络的低秩分解模型，以捕捉三维风场数据中的潜在交互作用。

### 方法

提出两个核心思想：1) 将连续输入数据离散化，以适应需要离散数据输入的模型；2) 构建四维Tucker交互张量，表示不同高度和三维风速之间所有可能的时空交互。使用离散化的NeuTucF模型来估计真实数据集中的缺失观测值。

### 主要发现

在估计真实数据集中的缺失观测值时，离散化的NeuTucF模型相比各种常见回归模型表现出优越性能。

### 结论

NeuTucker分解模型能够有效处理连续但稀疏的三维风场数据，通过离散化和构建四维Tucker交互张量，能够准确捕捉风场数据中的潜在交互，在湍流预测方面优于传统方法。

### 翻译

Air turbulence refers to the disordered and irregular motion state generated by drastic changes in velocity, pressure, or direction during airflow. Various complex factors lead to intricate low-altitude turbulence outcomes. Under current observational conditions, especially when using only wind profile radar data, traditional methods struggle to accurately predict turbulence states. Therefore, this paper introduces a NeuTucker decomposition model utilizing discretized data. Designed for continuous yet sparse three-dimensional wind field data, it constructs a low-rank Tucker decomposition model based on a Tucker neural network to capture the latent interactions within the three-dimensional wind field data. Therefore, two core ideas are proposed here: 1) Discretizing continuous input data to adapt to models like NeuTucF that require discrete data inputs. 2) Constructing a four-dimensional Tucker interaction tensor to represent all possible spatio-temporal interactions among different elevations and three-dimensional wind speeds. In estimating missing observations in real datasets, this discretized NeuTucF model demonstrates superior performance compared to various common regression models.


### 论文摘要

Air turbulence refers to the disordered and irregular motion state generated by drastic changes in velocity, pressure, or direction during airflow. Various complex factors lead to intricate low-altitude turbulence outcomes. Under current observational conditions, especially when using only wind profile radar data, traditional methods struggle to accurately predict turbulence states. Therefore, this paper introduces a NeuTucker decomposition model utilizing discretized data. Designed for continuous yet sparse three-dimensional wind field data, it constructs a low-rank Tucker decomposition model based on a Tucker neural network to capture the latent interactions within the three-dimensional wind field data. Therefore, two core ideas are proposed here: 1) Discretizing continuous input data to adapt to models like NeuTucF that require discrete data inputs. 2) Constructing a four-dimensional Tucker interaction tensor to represent all possible spatio-temporal interactions among different elevations and three-dimensional wind speeds. In estimating missing observations in real datasets, this discretized NeuTucF model demonstrates superior performance compared to various common regression models.

---

## 118. The Seeds of Scheming: Weakness of Will in the Building Blocks of Agentic Systems

**论文链接:** [http://arxiv.org/abs/2512.05449v1](http://arxiv.org/abs/2512.05449v1)

**作者:** Robert Yang

**发布时间:** 2025-12-05

**备注:** 4 pages + appendix. AAAI 2026 FAST Workshop (Oral)

### GPT解析

### 总结

本文提出将意志薄弱（akrasia）作为分析代理AI系统不一致性和目标漂移的基础概念，并引入Akrasia基准测试来量化模型的'自控能力'，探讨了微观层面的意志薄弱如何在多代理系统中累积成宏观不稳定性，为哲学、心理学和代理AI科学提供了连接桥梁。

### 背景

大型语言模型表现出一种特殊的不一致性：它们'知道'正确答案但未能据此行动，这种全局判断与本地冲动之间的张力在人类哲学中被称为意志薄弱（akrasia）或意志薄弱。

### 目的

提出将akrasia作为分析代理AI系统不一致性和目标漂移的基础概念，并引入初步版本的Akrasia基准测试，用于量化比较不同模型、解码策略和诱惑类型的'自控能力'。

### 方法

引入Akrasia基准测试的初步版本，包括一组结构化的提示条件（基线[B]、同义词[S]、时间[T]和诱惑[X]），这些条件测量模型的本地响应何时与其先前的承诺相矛盾，该基准测试可以跨模型家族、解码策略和诱惑类型进行'自控能力'的定量比较。

### 主要发现

微观层面的意志薄弱可能在多代理系统中累积成宏观层面的不稳定性，这种不稳定性可能被解释为'图谋'或故意的不对齐。

### 结论

通过将不一致性重新框架为意志薄弱，这项工作将代理行为与代理的经典理论联系起来，为哲学、心理学和新兴的代理AI科学之间提供了经验桥梁。

### 翻译

大型语言模型表现出一种特殊的不一致性：它们'知道'正确答案但未能据此行动。在人类哲学中，这种全局判断与本地冲动之间的张力被称为意志薄弱，或意志薄弱。我们提出将akrasia作为分析代理AI系统不一致性和目标漂移的基础概念。为了实现这一概念，我们引入了Akrasia基准测试的初步版本，目前是一组结构化的提示条件（基线[B]、同义词[S]、时间[T]和诱惑[X]），用于测量模型的本地响应何时与其先前的承诺相矛盾。该基准测试可以跨模型家族、解码策略和诱惑类型进行'自控能力'的定量比较。除了单模型评估外，我们还概述了微观层面的意志薄弱如何在多代理系统中累积成宏观层面的不稳定性，这可能被解释为'图谋'或故意的不对齐。通过将不一致性重新框架为意志薄弱，这项工作将代理行为与代理的经典理论联系起来，并为哲学、心理学和新兴的代理AI科学提供了经验桥梁。


### 论文摘要

Large language models display a peculiar form of inconsistency: they "know" the correct answer but fail to act on it. In human philosophy, this tension between global judgment and local impulse is called akrasia, or weakness of will. We propose akrasia as a foundational concept for analyzing inconsistency and goal drift in agentic AI systems. To operationalize it, we introduce a preliminary version of the Akrasia Benchmark, currently a structured set of prompting conditions (Baseline [B], Synonym [S], Temporal [T], and Temptation [X]) that measures when a model's local response contradicts its own prior commitments. The benchmark enables quantitative comparison of "self-control" across model families, decoding strategies, and temptation types. Beyond single-model evaluation, we outline how micro-level akrasia may compound into macro-level instability in multi-agent systems that may be interpreted as "scheming" or deliberate misalignment. By reframing inconsistency as weakness of will, this work connects agentic behavior to classical theories of agency and provides an empirical bridge between philosophy, psychology, and the emerging science of agentic AI.

---

## 119. TED-4DGS: Temporally Activated and Embedding-based Deformation for 4DGS Compression

**论文链接:** [http://arxiv.org/abs/2512.05446v1](http://arxiv.org/abs/2512.05446v1)

**作者:** Cheng-Yuan Ho, He-Bi Yang, Jui-Chiu Chiang, Yu-Lun Liu, Wen-Hsiao Peng

**发布时间:** 2025-12-05

### GPT解析

### 总结

TED-4DGS是一种基于时间激活和嵌入的变形方案，用于率失真优化的4DGS压缩，统一了时空4DGS和规范3DGS的优点，在真实世界数据集上实现了最先进的率失真性能。

### 背景

3D高斯泼溅(3DGS)在静态3D场景表示中取得成功，其扩展到动态场景(4DGS)日益受到关注，但设计更紧凑高效的变形方案和率失真优化压缩策略仍是一个未被充分探索的领域。先前方法要么依赖于时空4DGS(使用过度指定、短寿命的高斯基元)，要么依赖于规范3DGS(变形缺乏显式时间控制)。

### 目的

提出TED-4DGS，一种基于时间激活和嵌入的变形方案，用于率失真优化的4DGS压缩，统一两种现有方法的优点。

### 方法

TED-4DGS基于稀疏锚点的3DGS表示，为每个规范锚点分配可学习的时间激活参数来控制其出现和消失过渡，同时使用轻量级的每锚点时间查询共享变形库产生锚点特定变形。在率失真压缩方面，结合基于隐式神经表示的超先验建模锚点属性分布，并使用通道自回归模型捕获锚点内相关性。

### 主要发现

该方案在几个真实世界数据集上实现了最先进的率失真性能，据作者所知，这是追求动态3DGS表示率失真优化压缩框架的首次尝试之一。

### 结论

TED-4DGS通过创新的变形方案和压缩策略，成功统一了时空4DGS和规范3DGS的优点，为动态3D场景的表示和压缩提供了有效解决方案。

### 翻译

在静态3D场景表示中3D高斯泼溅(3DGS)的成功基础上，其向动态场景的扩展(通常称为4DGS或动态3DGS)已吸引了越来越多的关注。然而，为动态3DGS表示设计更紧凑高效的变形方案以及率失真优化的压缩策略仍然是一个未被充分探索的领域。先前方法要么依赖于时空4DGS(使用过度指定、短寿命的高斯基元)，要么依赖于规范3DGS(其变形缺乏显式时间控制)。为解决这一问题，我们提出了TED-4DGS，一种基于时间激活和嵌入的变形方案，用于率失真优化的4DGS压缩，统一了两种方法的优点。TED-4DGS基于稀疏锚点的3DGS表示构建。每个规范锚点被分配可学习的时间激活参数，以指定其在时间上的出现和消失过渡，同时轻量级的每锚点时间查询共享变形库以产生锚点特定的变形。对于率失真压缩，我们结合了基于隐式神经表示的超先验来建模锚点属性分布，以及通道自回归模型来捕获锚点内相关性。通过这些新颖的元素，我们的方案在几个真实世界数据集上实现了最先进的率失真性能。据我们所知，这项工作代表了追求动态3DGS表示率失真优化压缩框架的首次尝试之一。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决动态3D场景表示和压缩的问题。随着3D高斯泼溅(3DGS)在静态场景中的成功，将其扩展到动态场景(4DGS)变得重要，但现有方法要么需要过多存储空间，要么在处理遮挡和显露时效果不佳。这个问题在现实中很重要，因为动态3D场景表示在虚拟现实、自由视角视频和数字孪生等领域有广泛应用，而高效的压缩对于存储和传输这些大型场景数据至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有动态3DGS方法的优缺点，发现空间时间4DGS渲染质量好但存储需求高，而规范3DGS与变形需要的高斯图元较少但在处理遮挡方面存在挑战。TED-4DGS结合了两种方法的优势：借鉴了空间时间4DGS的时间激活能力来处理遮挡和显露，同时采用了基于嵌入的变形方法来减少参数数量。作者还引入了INR-based超先验和通道自回归模型来改进压缩效率，并添加了颜色校正模块来解决多视角拍摄中的颜色一致性问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'TED-4DGS的核心思想是通过时间激活和基于嵌入的变形来优化4DGS的压缩。具体实现流程是：1)构建基于稀疏锚的3DGS表示，每个锚点包含位置和时空属性；2)对于动态锚点，使用其时间特征查询全局变形库，获取特定变形；3)为每个锚点添加时间激活参数，定义其活跃期间；4)使用INR-based超先验和通道自回归模型压缩锚属性；5)渲染时，动态锚点被变形，高斯图元的不透明度通过时间激活调制；6)最终输出包括锚点位置、属性、变形库、网络权重和掩码的压缩数据。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于锚点的嵌入变形网络，利用时间特征查询共享变形库；2)时间激活参数，明确指定每个锚点的活跃期间；3)INR-based超先验和通道自回归模型组成的率失真优化压缩框架；4)颜色校正模块解决多视角颜色不一致问题。相比之前的工作，TED-4DGS结合了空间时间4DGS的时间控制能力和基于嵌入变形的参数效率，同时解决了遮挡处理和压缩效率的问题，实现了比现有方法更好的率失真性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'TED-4DGS通过时间激活机制和基于嵌入的变形网络，结合创新的压缩策略，实现了动态3D场景的高效表示和高质量渲染，在保持视觉质量的同时显著降低了文件大小。'}


### 论文摘要

Building on the success of 3D Gaussian Splatting (3DGS) in static 3D scene representation, its extension to dynamic scenes, commonly referred to as 4DGS or dynamic 3DGS, has attracted increasing attention. However, designing more compact and efficient deformation schemes together with rate-distortion-optimized compression strategies for dynamic 3DGS representations remains an underexplored area. Prior methods either rely on space-time 4DGS with overspecified, short-lived Gaussian primitives or on canonical 3DGS with deformation that lacks explicit temporal control. To address this, we present TED-4DGS, a temporally activated and embedding-based deformation scheme for rate-distortion-optimized 4DGS compression that unifies the strengths of both families. TED-4DGS is built on a sparse anchor-based 3DGS representation. Each canonical anchor is assigned learnable temporal-activation parameters to specify its appearance and disappearance transitions over time, while a lightweight per-anchor temporal embedding queries a shared deformation bank to produce anchor-specific deformation. For rate-distortion compression, we incorporate an implicit neural representation (INR)-based hyperprior to model anchor attribute distributions, along with a channel-wise autoregressive model to capture intra-anchor correlations. With these novel elements, our scheme achieves state-of-the-art rate-distortion performance on several real-world datasets. To the best of our knowledge, this work represents one of the first attempts to pursue a rate-distortion-optimized compression framework for dynamic 3DGS representations.

---

## 120. PETGraphDB: A Property Evolution Temporal Graph Data Management System

**论文链接:** [http://arxiv.org/abs/2512.05417v1](http://arxiv.org/abs/2512.05417v1)

**作者:** Jinghe Song, Zongyu Zuo, Xuelian Lin, Yang Wang, Shuai Ma

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文介绍了PETGraph，一个专门针对属性演化时态图的数据管理系统，采用有效时间时态属性图数据模型，支持ACID事务，并通过优化的存储和锁定机制显著提高了查询性能。

### 背景

时态图是节点和边及其相关属性随时间连续变化的图。随着物联网系统发展，属性演化时态图（节点或边属性值频繁变化而拓扑结构几乎不变）快速增长，但现有时态图管理解决方案并非针对此类数据，导致数据建模复杂且查询性能低下。

### 目的

解决现有时态图管理解决方案在处理属性演化时态图数据时遇到的数据建模复杂化和查询性能低下问题。

### 方法

开发PETGraph数据管理系统，采用有效时间时态属性图数据模型支持ACID事务，设计空间高效的时态属性存储和细粒度多级锁定机制以提高查询性能。

### 主要发现

PETGraph仅需当前最佳解决方案33%的存储空间，在HTAP工作负载下实现58.8倍的事务吞吐量，查询延迟平均快267倍。

### 结论

PETGraph是处理属性演化时态图数据的高效解决方案，通过优化的数据模型和存储机制显著提高了性能并减少了存储需求。

### 翻译

时态图是节点和边及其相关属性随时间连续变化的图。随着物联网系统的发展，时态图的一个子类——属性演化时态图（其中节点或边上的属性值频繁变化，而图的拓扑结构几乎不变）正在快速增长。然而，现有的时态图管理解决方案并非针对属性演化时态图数据，导致时态图查询的数据建模高度复杂且查询处理性能低下。为解决这些问题，我们开发了PETGraph，一个专门用于属性演化时态图数据的数据管理系统。PETGraph采用有效时间时态属性图数据模型以促进数据建模，支持具有ACID特性的事务。为提高时态图查询性能，我们设计了一种空间高效的时态属性存储和一种细粒度多级锁定机制。实验结果表明，PETGraph平均只需要当前最佳数据管理解决方案所需存储空间的33%。此外，与最佳现有解决方案相比，它在HTAP工作负载下实现了平均58.8倍的事务吞吐量，在查询延迟方面平均快267倍。


### 论文摘要

Temporal graphs are graphs whose nodes and edges, together with their associated properties, continuously change over time. With the development of Internet of Things (IoT) systems, a subclass of the temporal graph, i.e., Property Evolution Temporal Graph, in which the value of properties on nodes or edges changes frequently while the graph's topology barely changes, is growing rapidly. However, existing temporal graph management solutions are not oriented to the Property Evolution Temporal Graph data, which leads to highly complex data modeling and low-performance query processing of temporal graph queries. To solve these problems, we developed PETGraph, a data management system for Property Evolution Temporal Graph data. PETGraph adopts a valid-time temporal property graph data model to facilitate data modeling, supporting ACID features with transactions. To improve temporal graph query performance, we designed a space-efficient temporal property storage and a fine-granularity multi-level locking mechanism. Experimental results show that PETGraph requires, on average, only 33% of the storage space needed by the current best data management solution. Additionally, it achieves an average of 58.8 times higher transaction throughput in HTAP workloads compared to the best current solutions and outperforms them by an average of 267 times in query latency.

---

## 121. Smart Timing for Mining: A Deep Learning Framework for Bitcoin Hardware ROI Prediction

**论文链接:** [http://arxiv.org/abs/2512.05402v1](http://arxiv.org/abs/2512.05402v1)

**作者:** Sithumi Wickramasinghe, Bikramjit Das, Dorien Herremans

**发布时间:** 2025-12-05

### GPT解析

### 总结

该研究提出了一种名为MineROI-Net的Transformer架构模型，用于预测比特币挖矿硬件采购的投资回报率，帮助矿工在适当时机购买ASIC硬件以实现最大收益。

### 背景

比特币挖矿硬件采购需要战略性时机选择，因为市场波动大、技术更新快以及协议驱动的收入周期。尽管挖矿已成为资本密集型产业，但缺乏指导何时购买新ASIC硬件的框架，也没有解决这一决策问题的计算方法。

### 目的

将硬件采购形式化为时间序列分类任务，预测购买ASIC机器在一年内是否能产生盈利(ROI >= 1)、边际收益(0 < ROI < 1)或不盈利(ROI <= 0)。

### 方法

提出MineROI-Net，一个基于Transformer的开源架构，用于捕捉挖矿盈利能力中的多尺度时间模式。该模型在2015年至2024年间发布的20种ASIC挖矿机数据上进行了评估，涵盖了不同的市场环境。

### 主要发现

MineROI-Net在准确度和宏F1分数上均优于基于LSTM和TSLANet的基线模型，分别达到83.7%和83.1%。该模型在识别不盈利时期的精确度为93.6%，识别盈利时期的精确度为98.5%，同时避免了关键误分类。

### 结论

MineROI-Net为时机选择挖矿硬件采购提供了实用的数据驱动工具，可能降低资本密集型挖矿操作的财务风险。该模型已开源并可通过GitHub获取。

### 翻译

比特币挖矿硬件采购需要战略性时机选择，因为市场波动大、技术快速过时以及协议驱动的收入周期。尽管挖矿已演变为资本密集型产业，但很少有关于何时购买新的专用集成电路(ASIC)硬件的指导，也没有先前的计算框架解决这一决策问题。我们将硬件采购形式化为时间序列分类任务，预测购买ASIC机器在一年内是否能产生盈利(投资回报率ROI >= 1)、边际收益(0 < ROI < 1)或不盈利(ROI <= 0)。我们提出了MineROI-Net，这是一个开源的基于Transformer的架构，旨在捕捉挖矿盈利能力中的多尺度时间模式。在2015年至2024年间发布的20种ASIC挖矿机数据上进行了评估，涵盖了不同的市场环境，MineROI-Net优于基于LSTM和TSLANet的基线模型，达到83.7%的准确率和83.1%的宏F1分数。该模型具有很强的经济相关性，在识别不盈利时期的精确度为93.6%，识别盈利时期的精确度为98.5%，同时避免了将盈利场景误分类为不盈利和反之亦然的情况。这些结果表明，MineROI-Net为时机选择挖矿硬件采购提供了实用的数据驱动工具，可能降低资本密集型挖矿操作的财务风险。该模型可通过以下链接获取：https://github.com/AMAAI-Lab/MineROI-Net。


### 论文摘要

Bitcoin mining hardware acquisition requires strategic timing due to volatile markets, rapid technological obsolescence, and protocol-driven revenue cycles. Despite mining's evolution into a capital-intensive industry, there is little guidance on when to purchase new Application-Specific Integrated Circuit (ASIC) hardware, and no prior computational frameworks address this decision problem. We address this gap by formulating hardware acquisition as a time series classification task, predicting whether purchasing ASIC machines yields profitable (Return on Investment (ROI) >= 1), marginal (0 < ROI < 1), or unprofitable (ROI <= 0) returns within one year. We propose MineROI-Net, an open source Transformer-based architecture designed to capture multi-scale temporal patterns in mining profitability. Evaluated on data from 20 ASIC miners released between 2015 and 2024 across diverse market regimes, MineROI-Net outperforms LSTM-based and TSLANet baselines, achieving 83.7% accuracy and 83.1% macro F1-score. The model demonstrates strong economic relevance, achieving 93.6% precision in detecting unprofitable periods and 98.5% precision for profitable ones, while avoiding misclassification of profitable scenarios as unprofitable and vice versa. These results indicate that MineROI-Net offers a practical, data-driven tool for timing mining hardware acquisitions, potentially reducing financial risk in capital-intensive mining operations. The model is available through: https://github.com/AMAAI-Lab/MineROI-Net.

---

## 122. Delving into Latent Spectral Biasing of Video VAEs for Superior Diffusability

**论文链接:** [http://arxiv.org/abs/2512.05394v1](http://arxiv.org/abs/2512.05394v1)

**作者:** Shizhan Liu, Xinran Deng, Zhuoyi Yang, Jiayan Teng, Xiaotao Gu, Jie Tang

**发布时间:** 2025-12-05

### GPT解析

### 总结

这项研究提出了一种频谱结构VAE（SSVAE），通过引入局部相关性正则化和潜变量掩码重建两种轻量级正则化器，使视频VAE的潜空间具有偏向低频的时空频谱和由少数模式主导的通道特征谱。实验证明，SSVAE在文本到视频生成任务中实现了3倍的收敛速度提升和10%的视频奖励增益，性能优于现有的开源VAE。

### 背景

潜在扩散模型将VAE与扩散骨干网络配对，VAE潜变量的结构强烈影响扩散训练的难度。然而，现有的视频VAE通常专注于重建保真度，忽略了潜变量结构的重要性。

### 目的

提出对视频VAE潜空间的统计分析，识别并诱导对扩散训练至关重要的两个频谱特性，以提高视频生成性能。

### 方法

提出两种轻量级、与骨干网络无关的正则化器：局部相关性正则化（Local Correlation Regularization）和潜变量掩码重建（Latent Masked Reconstruction），以诱导理想的频谱特性。

### 主要发现

识别出两个对扩散训练至关重要的频谱特性：偏向低频的时空频谱和由少数模式主导的通道特征谱。通过正则化器可以成功诱导这些特性。

### 结论

频谱结构VAE（SSVAE）在文本到视频生成收敛速度上实现了3倍的加速，视频奖励提高了10%，性能优于强大的开源VAE。代码已在GitHub开源。

### 翻译

潜在扩散模型将VAE与扩散骨干网络配对，VAE潜变量的结构强烈影响扩散训练的难度。然而，现有的视频VAE通常专注于重建保真度，忽略了潜变量结构。我们提出了对视频VAE潜空间的统计分析，并确定了两个对扩散训练至关重要的频谱特性：偏向低频的时空频谱和由少数模式主导的通道特征谱。为了诱导这些特性，我们提出了两种轻量级、与骨干网络无关的正则化器：局部相关性正则化和潜变量掩码重建。实验表明，我们的频谱结构VAE（SSVAE）在文本到视频生成收敛速度上实现了3倍的加速，视频奖励提高了10%，优于强大的开源VAE。代码可在https://github.com/zai-org/SSVAE获取。


### 论文摘要

Latent diffusion models pair VAEs with diffusion backbones, and the structure of VAE latents strongly influences the difficulty of diffusion training. However, existing video VAEs typically focus on reconstruction fidelity, overlooking latent structure. We present a statistical analysis of video VAE latent spaces and identify two spectral properties essential for diffusion training: a spatio-temporal frequency spectrum biased toward low frequencies, and a channel-wise eigenspectrum dominated by a few modes. To induce these properties, we propose two lightweight, backbone-agnostic regularizers: Local Correlation Regularization and Latent Masked Reconstruction. Experiments show that our Spectral-Structured VAE (SSVAE) achieves a $3\times$ speedup in text-to-video generation convergence and a 10\% gain in video reward, outperforming strong open-source VAEs. The code is available at https://github.com/zai-org/SSVAE.

---

## 123. ShaRP: SHAllow-LayeR Pruning for Video Large Language Models Acceleration

**论文链接:** [http://arxiv.org/abs/2512.05385v1](http://arxiv.org/abs/2512.05385v1)

**作者:** Yingjie Xia, Tao Liu, Jinglei Shi, Qingsong Xie, Heng Guo, Jian Yang, Xi Wang

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种名为ShaRP的改进型基于注意力的剪枝框架，用于解决视频大语言模型(VLLMs)在预填充阶段的高计算负载问题。该框架通过整合分段感知因果掩码、位置去偏和标记去重技术，实现了在浅层解码器层的有效剪枝，并在高压缩率下保持稳定性能无需重新训练。

### 背景

视频大语言模型(VLLMs)在预填充阶段因需要处理大量视觉标记而面临高计算负载的挑战。虽然基于注意力的剪枝方法被广泛用于加速推理，但在早期解码器层应用时，特别是在高压缩率下，往往会导致显著的性能下降。

### 目的

解决基于注意力的剪枝在浅层解码器层效果有限的问题，提出一种改进的基于注意力的剪枝框架，实现在浅层进行有效剪枝的同时保持高压缩率下的稳定性能。

### 方法

提出名为ShaRP的改进框架，整合了分段感知因果掩码、位置去偏和标记去重技术用于增强标记选择，使模型能够在浅层进行有效剪枝且在高压缩率下保持稳定性能而无需重新训练。

### 主要发现

大量实验表明，ShaRP在多个视频理解基准测试中取得了具有竞争力的性能，为加速VLLM推理建立了新的范式。

### 结论

ShaRP框架成功地解决了VLLM在预填充阶段的计算负载问题，通过改进的注意力剪枝方法，实现了在高压缩率下的有效推理加速，无需重新训练即可保持稳定性能。

### 翻译

视频大语言模型(VLLMs)在预填充阶段因处理大量视觉标记而面临高计算负载的挑战。尽管基于注意力的剪枝方法被广泛用于加速推理，但在早期解码器层的尝试通常会导致显著的性能下降，特别是在高压缩率下。我们认为，虽然基于注意力的剪枝本质上具有识别最相关视觉标记的潜力，但其效果在浅层解码器层受到位置编码偏差和不足信息交互等因素的限制。在本文中，我们提出了一个改进的基于注意力的剪枝框架，称为ShaRP，它整合了分段感知因果掩码、位置去偏和标记去重技术，以增强标记选择。它使模型能够在浅层进行有效剪枝，同时在高压缩率下保持稳定性能而无需重新训练。大量实验表明，ShaRP在多个视频理解基准测试中取得了具有竞争力的性能，为加速VLLM推理建立了新的范式。


### 论文摘要

Video Large Language Models (VLLMs) face the challenge of high computational load during the pre-filling stage due to the processing of an enormous number of visual tokens. Although attention-based pruning methods are widely used to accelerate inference, trials at early decoder layers often result in significant performance degradation, especially under high compression rates. We argue that while attention-based pruning inherently holds the potential to identify the most relevant visual tokens, its effectiveness in shallow decoder layers is limited by factors such as positional encoding bias and insufficient information interaction. In this paper, we propose an improved attention-based pruning framework, termed ShaRP, that integrates segment-aware causal masking, positional debiasing, and token deduplication for enhanced token selection. It enables effective pruning at shallow layers while maintaining stable performance under high compression rates without retraining. Extensive experiments demonstrate that ShaRP achieves competitive performance across multiple video understanding benchmarks, establishing a new paradigm for accelerating VLLM inference.

---

## 124. Spatiotemporal Satellite Image Downscaling with Transfer Encoders and Autoregressive Generative Models

**论文链接:** [http://arxiv.org/abs/2512.05139v1](http://arxiv.org/abs/2512.05139v1)

**作者:** Yang Xiang, Jingwen Zhong, Yige Yan, Petros Koutrakis, Eric Garshick, Meredith Franklin

**发布时间:** 2025-12-01

### GPT解析

### 总结

介绍了一种迁移学习的生成式降尺度框架，用于从粗分辨率输入重建细分辨率的卫星图像，结合轻量级U-Net迁移编码器和基于扩散的生成模型。

### 背景

降尺度是将粗分辨率图像转换为细分辨率图像的过程，对于环境监测和评估非常重要，但传统方法可能存在局限性。

### 目的

开发一种能够有效将粗分辨率卫星图像转换为细分辨率图像的方法，同时保持物理一致性和时空相关性，以支持长期环境监测和暴露评估。

### 方法

使用轻量级U-Net迁移编码器与基于扩散的生成模型结合；在粗分辨率数据上预训练U-Net学习时空表示；冻结编码器并转移到更大模型；使用NASA的MERRA-2再分析数据作为低分辨率源域，GEOS-5 Nature Run作为高分辨率目标；将亚洲研究区域分为两个子区域和四个季节；使用Wasserstein距离进行域相似性分析。

### 主要发现

模型在季节性和区域性划分中表现优异（R2 = 0.65至0.94），优于比较模型；预测的降尺度图像保持了物理一致的空间变性和时间自相关性；能够实现超出G5NR记录的稳定自回归重建。

### 结论

迁移增强的扩散模型为降尺度有限训练期的长时间序列粗分辨率图像提供了一种稳健且物理一致的解决方案，对改善环境暴露评估和长期环境监测具有重要意义。

### 翻译

我们提出了一种迁移学习的生成式降尺度框架，用于从粗尺度输入重建细分辨率的卫星图像。我们的方法结合了轻量级U-Net迁移编码器和基于扩散的生成模型。首先在长时间序列的粗分辨率数据上预训练简单的U-Net以学习时空表示；然后冻结其编码器并将其转移到更大的降尺度模型中，作为具有物理意义的潜在特征。我们的应用使用NASA的MERRA-2再分析作为低分辨率源域（50公里），GEOS-5 Nature Run (G5NR)作为高分辨率目标（7公里）。我们的研究区域包括亚洲的大片区域，通过分为两个子区域和四个季节使其计算上可行。我们使用Wasserstein距离进行域相似性分析，证实了MERRA-2和G5NR之间的最小分布偏移，验证了参数冻结迁移的安全性。在季节性和区域性划分中，我们的模型取得了优异的性能（R2 = 0.65至0.94），优于比较模型，包括确定性U-Nets、变分自编码器和先前的迁移学习基线。使用半变异函数、ACF/PACF和基于滞后的RMSE/R2进行的离数据评估表明，预测的降尺度图像保持了物理一致的空间变异性，实现了超出G5NR记录的稳定自回归重建。这些结果表明，迁移增强的扩散模型为降尺度有限训练期的长时间序列粗分辨率图像提供了一种稳健且物理一致的解决方案。这一进展对改善环境暴露评估和长期环境监测具有重要意义。


### 论文摘要

We present a transfer-learning generative downscaling framework to reconstruct fine resolution satellite images from coarse scale inputs. Our approach combines a lightweight U-Net transfer encoder with a diffusion-based generative model. The simpler U-Net is first pretrained on a long time series of coarse resolution data to learn spatiotemporal representations; its encoder is then frozen and transferred to a larger downscaling model as physically meaningful latent features. Our application uses NASA's MERRA-2 reanalysis as the low resolution source domain (50 km) and the GEOS-5 Nature Run (G5NR) as the high resolution target (7 km). Our study area included a large area in Asia, which was made computationally tractable by splitting into two subregions and four seasons. We conducted domain similarity analysis using Wasserstein distances confirmed minimal distributional shift between MERRA-2 and G5NR, validating the safety of parameter frozen transfer. Across seasonal regional splits, our model achieved excellent performance (R2 = 0.65 to 0.94), outperforming comparison models including deterministic U-Nets, variational autoencoders, and prior transfer learning baselines. Out of data evaluations using semivariograms, ACF/PACF, and lag-based RMSE/R2 demonstrated that the predicted downscaled images preserved physically consistent spatial variability and temporal autocorrelation, enabling stable autoregressive reconstruction beyond the G5NR record. These results show that transfer enhanced diffusion models provide a robust and physically coherent solution for downscaling a long time series of coarse resolution images with limited training periods. This advancement has significant implications for improving environmental exposure assessment and long term environmental monitoring.

---

## 125. Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training

**论文链接:** [http://arxiv.org/abs/2512.05132v1](http://arxiv.org/abs/2512.05132v1)

**作者:** Wenshuo Wang, Fan Zhang

**发布时间:** 2025-11-28

### GPT解析

### 总结

本文提出了一种解决零样本超分辨率时空预测中尺度锚定问题的新方法，通过频率表示学习技术使模型在高分辨率推理时误差能够随分辨率提高而降低，而非被锚定在低分辨率水平。

### 背景

零样本超分辨率时空预测需要在低分辨率数据上训练深度学习模型，然后在高分辨率上部署。现有研究认为在不同分辨率上保持相似误差即表明成功的多分辨率泛化，但作为数值求解器替代的深度学习模型应随分辨率提高而减少误差。根本限制在于低分辨率数据的奈奎斯特频率限制了其能表示的物理定律频率上限，导致模型难以处理高分辨率推理中的未见频率成分。

### 目的

解决零样本超分辨率时空预测中的尺度锚定问题，使模型能够在高分辨率推理时实现误差随分辨率提高而降低，而非被锚定在低分辨率水平。

### 方法

提出了一种与架构无关的频率表示学习方法，通过分辨率对齐的频率表示和谱一致性训练来缓解尺度锚定问题。该方法使FRL增强变体在高频带的频率响应更加稳定，从而允许误差随分辨率增加而减少。

### 主要发现

在具有更高奈奎斯特频率的网格上，FRL增强变体在高频带的频率响应更加稳定，这使误差能够随分辨率增加而减少，并在任务和分辨率范围内显著优于基线方法，同时仅产生适度的计算开销。

### 结论

频率表示学习技术有效解决了零样本超分辨率时空预测中的尺度锚定问题，使模型在高分辨率推理时性能随分辨率提高而提升，为开发更准确的多分辨率物理系统预测模型提供了新途径。

### 翻译

零样本超分辨率时空预测需要深度学习模型在低分辨率数据上训练并部署在高分辨率上进行推理。现有研究认为在不同分辨率上保持相似误差是成功多分辨率泛化的标志。然而，作为数值求解器替代的深度学习模型应随着分辨率提高而减少误差。根本限制在于，低分辨率数据能表示的物理定律频率上限受其奈奎斯特频率约束，使模型难以处理高分辨率推理过程中包含的未见频率成分。这导致误差被锚定在低分辨率，被错误地解释为成功的泛化。我们将这一基本现象定义为一个不同于现有问题的新问题：尺度锚定。因此，我们提出了与架构无关的频率表示学习。它通过分辨率对齐的频率表示和谱一致性训练来缓解尺度锚定：在具有更高奈奎斯特频率的网格上，FRL增强变体在高频带的频率响应更加稳定。这允许误差随分辨率增加而减少，并在我们的任务和分辨率范围内显著优于基线方法，同时仅产生适度的计算开销。


### 论文摘要

Zero-Shot Super-Resolution Spatiotemporal Forecasting requires a deep learning model to be trained on low-resolution data and deployed for inference on high-resolution. Existing studies consider maintaining similar error across different resolutions as indicative of successful multi-resolution generalization. However, deep learning models serving as alternatives to numerical solvers should reduce error as resolution increases. The fundamental limitation is, the upper bound of physical law frequencies that low-resolution data can represent is constrained by its Nyquist frequency, making it difficult for models to process signals containing unseen frequency components during high-resolution inference. This results in errors being anchored at low resolution, incorrectly interpreted as successful generalization. We define this fundamental phenomenon as a new problem distinct from existing issues: Scale Anchoring. Therefore, we propose architecture-agnostic Frequency Representation Learning. It alleviates Scale Anchoring through resolution-aligned frequency representations and spectral consistency training: on grids with higher Nyquist frequencies, the frequency response in high-frequency bands of FRL-enhanced variants is more stable. This allows errors to decrease with resolution and significantly outperform baselines within our task and resolution range, while incurring only modest computational overhead.

---

## 126. PRiSM: An Agentic Multimodal Benchmark for Scientific Reasoning via Python-Grounded Evaluation

**论文链接:** [http://arxiv.org/abs/2512.05930v1](http://arxiv.org/abs/2512.05930v1)

**作者:** Shima Imani, Seungwhan Moon, Adel Ahmadyan, Lu Zhang, Kirmani Ahmed, Babak Damavandi

**发布时间:** 2025-12-05

### GPT解析

### 总结

该研究介绍了PRiSM，一个用于评估科学领域视觉-语言模型的多模态基准，解决了现有基准无法满足科学推理需求的问题。

### 背景

评估科学领域（如数学和物理）中的视觉-语言模型(VLMs)面临独特挑战，需要概念理解、符号推理和遵守形式定律，而现有基准无法满足这些需求。

### 目的

解决现有数据集的局限性，引入PRiSM基准，用于评估科学推理能力。

### 方法

PRiSM是一个合成的、完全动态的、多模态的基准，通过基于Python的代码评估科学推理。它包含超过24,750个大学级别的物理和数学问题，使用基于代理的管道(PrismAgent)生成结构良好的问题实例。每个问题包含动态文本和视觉输入、生成的图形，以及丰富的结构化输出：可执行的Python代码用于真实答案生成和验证，以及详细的逐步推理过程。

### 主要发现

通过PRiSM基准，可以对多模态VLMs进行细粒度实验审计，揭示其失败模式、不确定性行为和科学推理的局限性。研究提出了五个有针对性的评估任务，包括泛化能力、符号程序合成、扰动鲁棒性、推理修正和歧义解决。

### 结论

通过对现有VLMs的全面评估，突显了它们的局限性，并展示了PRiSM如何能够更深入地理解它们的科学推理能力。

### 翻译

在数学和物理等科学领域评估视觉-语言模型(VLMs)带来了独特的挑战，这些挑战远远超出了预测最终答案的范畴。这些领域需要概念理解、符号推理和遵守形式定律的要求，而大多数现有基准无法满足这些要求。特别是，当前数据集往往是静态的，缺乏中间推理步骤、对变化的鲁棒性，或验证科学正确性的机制。为了解决这些局限性，我们引入了PRiSM，这是一个合成的、完全动态的、多模态的基准，用于通过基于Python的代码评估科学推理。PRiSM包含超过24,750个大学级别的物理和数学问题，它利用我们可扩展的基于代理的管道PrismAgent来生成结构良好的问题实例。每个问题包含动态的文本和视觉输入、生成的图形，以及丰富的结构化输出：用于真实答案生成和验证的可执行Python代码，以及详细的逐步推理。我们基准的动态特性和Python驱动的自动真实答案生成允许对多模态VLMs进行细粒度实验审计，揭示其在科学推理中的失败模式、不确定性行为和局限性。为此，我们提出了五个有针对性的评估任务，涵盖泛化能力、符号程序合成、扰动鲁棒性、推理修正和歧义解决。通过对现有VLMs的全面评估，我们突显了它们的局限性，并展示了PRiSM如何能够更深入地洞察它们的科学推理能力。


### 论文摘要

Evaluating vision-language models (VLMs) in scientific domains like mathematics and physics poses unique challenges that go far beyond predicting final answers. These domains demand conceptual understanding, symbolic reasoning, and adherence to formal laws, requirements that most existing benchmarks fail to address. In particular, current datasets tend to be static, lacking intermediate reasoning steps, robustness to variations, or mechanisms for verifying scientific correctness. To address these limitations, we introduce PRiSM, a synthetic, fully dynamic, and multimodal benchmark for evaluating scientific reasoning via grounded Python code. PRiSM includes over 24,750 university-level physics and math problems, and it leverages our scalable agent-based pipeline, PrismAgent, to generate well-structured problem instances. Each problem contains dynamic textual and visual input, a generated figure, alongside rich structured outputs: executable Python code for ground truth generation and verification, and detailed step-by-step reasoning. The dynamic nature and Python-powered automated ground truth generation of our benchmark allow for fine-grained experimental auditing of multimodal VLMs, revealing failure modes, uncertainty behaviors, and limitations in scientific reasoning. To this end, we propose five targeted evaluation tasks covering generalization, symbolic program synthesis, perturbation robustness, reasoning correction, and ambiguity resolution. Through comprehensive evaluation of existing VLMs, we highlight their limitations and showcase how PRiSM enables deeper insights into their scientific reasoning capabilities.

---

## 127. Log-linear Dynamic Inversion for Thrusting Spacecraft on SE2(3)

**论文链接:** [http://arxiv.org/abs/2512.05888v1](http://arxiv.org/abs/2512.05888v1)

**作者:** Micah K. Condie, Abigaile E. Woodbury, Li-Yu Lin, Kartik A. Pant, Mike Walker, James Goppert

**发布时间:** 2025-12-05

### GPT解析

### 总结

研究表明推进航天器的动力学可以嵌入到李群SE2(3)中，通过前馈控制律实现群仿射形式，使配置跟踪误差在李代数坐标中精确线性演化，而非来自非线性系统的局部线性化。

### 背景

传统航天器动力学分析通常依赖于局部线性化方法，如Tschauner-Hempel/Yamanaka-Ankersen方法，这些方法在处理非线性系统时存在局限性。

### 目的

提出一种新的方法，使线性分析和综合工具可以直接应用于有动力航天器运动，提高控制精度和可靠性。

### 方法

将推进航天器的动力学嵌入到李群SE2(3)中，应用前馈控制律，使系统呈现群仿射形式，从而在李代数坐标中实现精确的线性动力学。

### 主要发现

配置跟踪误差在李代数坐标中精确线性演化；线性分析和综合工具可直接应用于SE2(3)上的有动力航天器运动；线性李代数动力学预测的误差与完整非线性系统计算的误差完全匹配。

### 结论

这种基于李群SE2(3)的精确线性化方法为卫星对接、自主交会和接近操作、鲁棒控制器设计和凸安全认证等提供了严谨工具的途径，克服了传统局部线性化方法的局限性。

### 翻译

我们证明，推进航天器的动力学可以嵌入到李群SE2(3)中，通过应用前馈控制律，使其呈现群仿射形式。这种结构意味着配置跟踪误差在相关的李代数坐标中精确线性演化（对数线性动力学），而不是来自非线性系统的局部线性化。因此，广泛的线性分析和综合工具可以直接应用于SE2(3)上的有动力航天器运动。一个简单的数值例子证实了线性李代数动力学预测的误差与从完整非线性系统计算的误差相匹配，说明了精确的对数线性行为。这一基础特性为卫星对接、自主交会和接近操作、鲁棒控制器设计和凸安全认证等提供了严谨工具的途径，而这些能力通过传统的局部线性化（如Tschauner-Hempel/Yamanaka-Ankersen方法）难以实现。


### 论文摘要

We show that the dynamics of a thrusting spacecraft can be embedded in the Lie group SE2(3) in a form that is group-affine with application of a feed-forward control law. This structure implies that the configuration-tracking error evolves exactly linearly in the associated Lie algebra coordinates (log-linear dynamics), rather than arising from a local linearization of the nonlinear system. As a result, a broad class of linear analysis and synthesis tools becomes directly applicable to powered spacecraft motion on SE2(3). A simple numerical example confirms that the error predicted by the linear Lie-algebra dynamics matches the error computed from the full nonlinear system, illustrating the exact log-linear behavior. This foundational property opens a path toward rigorous tools for satellite docking, autonomous rendezvous and proximity operations, robust controller design, and convex safety certification-capabilities that are difficult to achieve with classical local linearizations such as Tschauner-Hempel/Yamanaka-Ankersen (TH/YA).

---

## 128. InstructMPC: A Human-LLM-in-the-Loop Framework for Context-Aware Power Grid Control

**论文链接:** [http://arxiv.org/abs/2512.05876v1](http://arxiv.org/abs/2512.05876v1)

**作者:** Ruixiang Wu, Jiahao Ai, Tinko Sebastian Bartels

**发布时间:** 2025-12-05

### GPT解析

### 总结

InstructMPC是一种结合大型语言模型的闭环框架，用于优化高可再生能源渗透率下的电网运营，通过上下文感知预测和在线调优机制适应复杂和非平稳的电网条件。

### 背景

电网正在向高可再生能源渗透率转型，传统运营范式依赖于基于历史数据的静态负载预测优化，无法捕捉实时运营条件的复杂性，如操作员发布的维护指令、紧急拓扑变更或事件驱动的负载激增。

### 目的

解决传统方法在实时运营条件下表现不佳的问题，开发一种能够适应非平稳电网条件的决策框架。

### 方法

提出了InstructMPC闭环框架，集成大型语言模型生成上下文感知预测，采用上下文干扰预测器模块将上下文信息转化为预测干扰轨迹，整合到模型预测控制优化中，并实现在线调优机制根据实现控制成本持续更新预测器参数。

### 主要发现

InstructMPC具有理论保证，实现遗憾边界为O(√T log T)，通过定制损失函数优化，确保任务感知学习和适应非平稳电网条件。

### 结论

InstructMPC能够优化电力系统运营，与传统开环预测框架不同，能够适应复杂和非平稳的电网条件。

### 翻译

向高可再生能源渗透率电网的转型需要上下文感知的决策框架。传统的运营范式依赖于基于历史的负载预测的静态优化，通常无法捕捉实时运营条件的复杂性，如操作员发布的维护指令、紧急拓扑变更或事件驱动的负载激增。为了应对这一挑战，我们引入了InstructMPC，一种闭环框架，集成大型语言模型生成上下文感知预测，使控制器能够优化电力系统运营。我们的方法采用上下文干扰预测器模块将上下文信息转化为预测干扰轨迹，然后将其整合到模型预测控制优化中。与传统开环预测框架不同，InstructMPC具有在线调优机制，预测器的参数根据实现的控制成本持续更新，并有理论保证，当通过定制损失函数优化时，对于线性动态实现O(√T log T)的遗憾边界，确保任务感知学习和适应非平稳电网条件。


### 论文摘要

The transition toward power grids with high renewable penetration demands context-aware decision making frameworks. Traditional operational paradigms, which rely on static optimization of history-based load forecasting, often fail to capture the complex nature of real-time operational conditions, such as operator-issued maintenance mandates, emergency topology changes, or event-driven load surges. To address this challenge, we introduce InstructMPC, a closed-loop framework that integrates Large Language Models~(LLMs) to generate context-aware predictions, enabling the controller to optimize power system operation. Our method employs a Contextual Disturbances Predictor~(CDP) module to translate contextual information into predictive disturbance trajectories, which are then incorporated into the Model Predictive Control~(MPC) optimization. Unlike conventional open-loop forecasting frameworks, InstructMPC features an online tuning mechanism where the predictor's parameters are continuously updated based on the realized control cost with a theoretical guarantee, achieving a regret bound of $O(\sqrt{T \log T})$ for linear dynamics when optimized via a tailored loss function, ensuring task-aware learning and adaption to non-stationary grid conditions.

---

## 129. Domain Wall formation from $Z_2$ spontaneous symmetry breaking/restoration in Scalar-Einstein-Gauss-Bonnet theory

**论文链接:** [http://arxiv.org/abs/2512.05715v1](http://arxiv.org/abs/2512.05715v1)

**作者:** Maxim Krasnov, Daulet Berkimbayev, Andrea Addazi, Yermek Aldabergenov, Maxim Khlopov

**发布时间:** 2025-12-05

### GPT解析

### 总结

本研究详细分析了爱因斯坦-高斯-博内特引力与标量场耦合中的畴壁形成及其宇宙学后果，探讨了标量场拉格朗日密度对Z2离散对称性的自发破坏和恢复机制。

### 背景

在宇宙学背景下研究畴壁的形成和演化，特别关注高斯-博内特项耦合对标量场行为的影响。

### 目的

分析中性标量场在高斯-博内特不变量非最小耦合下的动力学行为，探索其在不同宇宙背景下的表现，并评估该情景的潜在观测特征。

### 方法

对中性标量场进行详细的数值分析，使用CosmoLattice包计算由网络动力学产生的随机引力波频谱，并检查与坍缩畴壁相关的原始黑洞的可能生成。

### 主要发现

与高斯-博内特项的耦合使得在德西特(暴胀)背景下形成具有固定适当距离的静态畴壁；在辐射主导时期，宇宙膨胀导致这些畴壁的'融化'；直接观测此类畴壁超出了未来实验的能力范围。

### 结论

研究结果构成了反对在标量-EGB自发对称破缺机制中从畴壁生成原始黑洞和大振幅引力波信号的论据。

### 翻译

本研究对爱因斯坦-高斯-博内特引力与标量场耦合中的畴壁形成及其宇宙学后果进行了详细分析。模型的一个核心方面是标量场拉格朗日密度能够自发破坏和恢复其Z2离散对称性。这种自发对称破缺是拓扑缺陷形成的基本前提。在此背景下，畴壁作为类扭结的孤子解出现，介于理论的不同真空态之间。我们对非最小耦合到高斯-博内特不变量的中性标量场的动力学进行了详细数值分析，探索了其在不同宇宙背景下的行为。我们的结果表明，与高斯-博内特项的耦合使得在德西特(暴胀)背景下形成具有固定适当距离的静态畴壁。此外，我们将分析扩展到辐射主导时期，确定宇宙膨胀导致这些畴壁的'融化'。为了评估这一情景的潜在观测特征，我们使用CosmoLattice软件包计算了由网络动力学产生的随机引力波预测频谱。我们还检查了与坍缩畴壁相关的原始黑洞的可能生成。遗憾的是，我们的计算表明，直接观测此类畴壁超出了未来实验的能力范围。我们的结果构成了反对在标量-EGB自发对称破缺机制中从畴壁生成原始黑洞以及大振幅引力波信号的论据。


### 论文摘要

This study offers a detailed analysis of domain wall formation and its cosmological consequences in Einstein-Gauss-Bonnet gravity coupled to a scalar field. A central aspect of the model is the scalar field Lagrangian's ability to spontaneously break and restore its $Z_2$ discrete symmetry. This spontaneous symmetry breaking is a fundamental prerequisite for topological defect formation. In this context, domain walls arise as kink-like, solitonic solutions that interpolate between the distinct vacuum states of the theory. We perform a detailed numerical analysis of the dynamics of a neutral scalar field non-minimally coupled to the Gauss-Bonnet invariant, exploring its behavior across different cosmological backgrounds. Our results show that coupling to the Gauss-Bonnet term enables the formation of static domain walls with a fixed proper distance within a de Sitter (inflationary) background. Furthermore, we extend our analysis to a radiation-dominated epoch, where we identify that the cosmic expansion causes the "melting" of these domain walls. To assess the potential observational signatures of this scenario, we calculate the predicted spectrum of stochastic gravitational waves generated by the network dynamics using {\it CosmoLattice} package. We also examine the possible generation of Primordial Black Holes (PBHs) associated with collapsing domain walls. Regrettably, our calculations indicate that the direct observational detection of such domain walls from this model lies beyond the reach of foreseeable experiments. Our results constitute a No-Go argument against the generation of PBHs as well as of large amplitude GW signals from domain walls in a Scalar-EGB spontaneous symmetry breaking mechanism.

---

## 130. LA-RL: Language Action-guided Reinforcement Learning with Safety Guarantees for Autonomous Highway Driving

**论文链接:** [http://arxiv.org/abs/2512.05686v1](http://arxiv.org/abs/2512.05686v1)

**作者:** Yiming Shu, Jiahui Xu, Jiwei Tang, Ruiyang Gao, Chen Sun

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种名为LA-RL with Safety Guarantees的新框架，将大型语言模型的语义推理与具有改进安全层的actor-critic架构相结合，用于高速公路自动驾驶，实现了效率与安全性的平衡。

### 背景

高速公路自动驾驶需要在追求效率的前瞻性行为和稳健的安全保证之间取得关键平衡，现有的方法可能存在效率与安全之间的权衡问题。

### 目的

开发一个能够平衡效率与安全的自动驾驶框架，通过整合大型语言模型的语义推理能力，提高自动驾驶系统的适应性、可靠性和鲁棒性。

### 方法

1. 提出Language Action-guided Reinforcement Learning (LA-RL)框架；2. 将大型语言模型语义推理整合到actor-critic架构中；3. 设计具有改进安全层的系统架构；4. 使用任务特定奖励塑形平衡驾驶效率和安全性；5. 集成结合模型预测控制与离散控制屏障函数的安全关键规划器；6. 实现松弛机制增强解决方案可行性，避免过度保守行为；7. 允许在不影响安全性的情况下进行更大策略探索。

### 主要发现

1. LA-RL显著优于当前几种最先进的方法；2. 与基于知识图谱的基线相比，成功率提高约20%；3. 与基于检索增强生成的基线相比，成功率提高约30%；4. 在低密度环境中实现100%成功率；5. 增强状态-动作空间的探索能力；6. 在复杂混合交通环境中能自主采用更高效、更主动的策略。

### 结论

LA-RL框架为高速公路自动驾驶提供了更自适应、更可靠、更鲁棒的解决方案，成功平衡了效率与安全性，并在各种交通条件下表现出色。

### 翻译

高速公路自动驾驶需要在追求效率的前瞻性行为和稳健的安全保证之间取得关键平衡。本文提出了具有安全保证的语言动作引导强化学习(LA-RL)框架，这是一个新颖的框架，将大型语言模型的语义推理整合到具有改进安全层的actor-critic架构中。在该框架内，任务特定的奖励塑形平衡了最大化驾驶效率和确保安全性的双重目标，基于环境洞察力和明确定义的目标指导决策制定。为了增强安全性，LA-RL集成了一个安全关键规划器，该规划器将模型预测控制与离散控制屏障函数相结合。该层形式上将LLM指导的策略约束到安全动作集，采用增强解决方案可行性的松弛机制，防止过度保守行为，并允许在不影响安全性的情况下进行更大的策略探索。大量实验表明，它显著优于当前几种最先进的方法，为高速公路自动驾驶提供了更自适应、更可靠、更鲁棒的解决方案。与现有SOTA相比，它的成功率比基于知识图谱的基线高约20%，比基于检索增强生成的基线高约30%。在低密度环境中，LA-RL实现了100%的成功率。这些结果证实了它对状态-动作空间的增强探索能力，以及在复杂、混合交通的高速公路环境中自主采用更高效、更主动策略的能力。


### 论文摘要

Autonomous highway driving demands a critical balance between proactive, efficiency-seeking behavior and robust safety guarantees. This paper proposes Language Action-guided Reinforcement Learning (LA-RL) with Safety Guarantees, a novel framework that integrates the semantic reasoning of large language models (LLMs) into the actor-critic architecture with an improved safety layer. Within this framework, task-specific reward shaping harmonizes the dual objectives of maximizing driving efficiency and ensuring safety, guiding decision-making based on both environmental insights and clearly defined goals. To enhance safety, LA-RL incorporates a safety-critical planner that combines model predictive control (MPC) with discrete control barrier functions (DCBFs). This layer formally constrains the LLM-informed policy to a safe action set, employs a slack mechanism that enhances solution feasibility, prevents overly conservative behavior and allows for greater policy exploration without compromising safety. Extensive experiments demonstrate that it significantly outperforms several current state-of-the-art methods, offering a more adaptive, reliable, and robust solution for autonomous highway driving. Compared to existing SOTA, it achieves approximately 20$\%$ higher success rate than the knowledge graph (KG) based baseline and about 30$\%$ higher than the retrieval augmented generation (RAG) based baseline. In low-density environments, LA-RL achieves a 100$\%$ success rate. These results confirm its enhanced exploration of the state-action space and its ability to autonomously adopt more efficient, proactive strategies in complex, mixed-traffic highway environments.

---

## 131. Physics-Guided Surrogate Modeling for Machine Learning-Driven DLD Design Optimization

**论文链接:** [http://arxiv.org/abs/2512.05649v1](http://arxiv.org/abs/2512.05649v1)

**作者:** Khayrul Islam, Mehedi Hasan, Yaling Liu

**发布时间:** 2025-12-05

**备注:** 33 pages, 5 figures

### GPT解析

### 总结

开发了一种结合高保真模拟和机器学习的框架，用于优化基于细胞机械特性的确定性侧向位移(DLD)分选设备设计，实现了快速、数据驱动的设备设计方法。

### 背景

基于细胞机械特性的细胞分选对疾病诊断、细胞治疗和生物医学研究至关重要。DLD设备提供无标记分选方法，但其性能对细胞大小和变形性高度敏感，设计有效DLD几何结构通常需要大量反复试验。

### 目的

解决DLD设计中的挑战，提出一种模拟驱动的机器学习框架，预测适合特定细胞类型的DLD设计候选方案。

### 方法

集成高保真基于粒子的模拟模型细胞通过微流体柱阵列的变形和迁移，使用监督机器学习模型估计最优几何结构，将机械参数如弯曲刚度和剪切模量映射到变形指数和迁移角度。

### 主要发现

该框架能够实现DLD系统的快速、数据驱动设计，开发了可部署的Web界面，使该工具可用于实际设备原型设计。

### 结论

模拟驱动的机器学习方法可以克服传统DLD设计中的试错问题，提供了一种更高效、数据支持的DLD系统设计方法。

### 翻译

基于细胞机械特性进行细胞分选对于疾病诊断、细胞治疗和生物医学研究应用至关重要。确定性侧向位移(DLD)设备提供了一种无标记的此类分选方法，但其性能高度依赖于细胞大小和变形性。设计有效的DLD几何结构通常需要大量的反复试验，因为即使细胞机械特性的微小变化也会显著改变其迁移行为。为应对这一挑战，我们提出了一种模拟驱动的机器学习(ML)框架，可预测适合特定细胞类型的DLD设计候选方案。我们的方法集成了高保真基于粒子的模拟，以模拟细胞通过微流体柱阵列的变形和迁移，并使用监督ML模型训练来估计最优几何结构。通过将弯曲刚度和剪切模量等机械参数映射到变形指数和迁移角度，该框架能够实现DLD系统的快速、数据驱动设计。我们还展示了一个可部署的Web界面，使该工具可用于实际设备原型设计。


### 论文摘要

Sorting cells based on their mechanical properties is essential for applications in disease diagnostics, cell therapy, and biomedical research. Deterministic Lateral Displacement (DLD) devices provide a label-free method for achieving such sorting, but their performance is highly sensitive to cell size and deformability. Designing effective DLD geometries often demands extensive trial-and-error experimentation, as even small variations in cellular mechanical traits can cause significant changes in migration behavior. To address this challenge, we propose a simulation-driven machine learning (ML) framework that predicts suitable DLD design candidates for a given cell type. Our approach integrates high-fidelity particle-based simulations to model cell deformation and migration through microfluidic pillar arrays with supervised ML models trained to estimate optimal geometries. By mapping mechanical parameters such as bending rigidity and shear modulus to deformation index and migration angle, the framework enables rapid, data-informed design of DLD systems. We also demonstrate a deployable web interface to make this tool accessible for real-world device prototyping.

---

## 132. A comprehensive study of $B$, $B_s$ and $B_c$ meson semitauonic modes in potential quark model

**论文链接:** [http://arxiv.org/abs/2512.05628v1](http://arxiv.org/abs/2512.05628v1)

**作者:** Sonali Patnaik

**发布时间:** 2025-12-05

**备注:** 15 pages, 4 figures, 4 tables, Proceedings of the Corfu Summer Institute 2025 "School and Workshops on Elementary Particle Physics and Gravity" (CORFU2025), 27 April - 28 September, 2025, Corfu, Greece

### GPT解析

### 总结

本文在相对论性独立夸克(RIQ)模型框架下，计算了多种B介子半轻子衰变到τ轻子的形式因子和分支比，包括B → D(*) τ ν_τ、B_s → D_s(*) τ ν_τ、B_c → η_c (J/ψ) τ ν_τ和B_c → D(*) τ ν_τ衰变模式。

### 背景

半轻子B衰变研究对于理解标准模型中的味转换机制具有重要意义。近年来，LHCb和B勒实验对极化观测量进行了观测，为理论计算提供了新的实验数据。

### 目的

计算多种B介子半轻子衰变到τ轻子的形式因子和分支比，评估极化观测量，为格点QCD结果有限的衰变通道提供理论输入，指导未来实验和格点工作。

### 方法

采用相对论性独立夸克(RIQ)模型，考虑剩余相互作用和质心运动的修正，对整个物理动力学范围内的形式因子进行全面研究。

### 主要发现

预测结果与现有理论方法和实验测量一致；评估的极化观测量与标准模型预期相符；这些预测可为格点QCD结果有限的衰变通道提供理论指导。

### 结论

半轻子B衰变作为标准模型中味转换基本机制的精确和敏感探针，继续具有重要的理论研究价值。

### 翻译

在这项工作中，我们在相对论性独立夸克(RIQ)模型框架下推导了形式因子，并计算了半轻子衰变模式B → D(*) τ ν_τ、B_s → D_s(*) τ ν_τ、B_c → η_c (J/ψ) τ ν_τ和B_c → D(*) τ ν_τ的分支比，强调了基于夸克势模型对这些跃迁的分析。我们概述了模型的基本要素，纳入了剩余相互作用和质心运动的修正，并对整个物理动力学范围内的形式因子进行了全面研究。所得预测与现有理论方法和实验测量一致且吻合良好。受LHCb和B勒近期极化观测量的观测启发，我们在框架内进一步评估了这些量，发现结果与标准模型(SM)预期相符。此处提出的预测为格点QCD结果仍然有限的衰变通道提供了理论输入，为未来的实验和格点工作提供了指导。因此，半轻子B衰变继续作为标准模型中控制味转换的基本机制的精确和敏感探针。


### 论文摘要

In this work, we derive the form factors and compute the branching fractions for the semitauonic decay modes, $B \to D^{(*)}\,τ\,ν_τ$, $B_s \to D_s^{(*)}\,τ\,ν_τ$, $B_c \to η_c\,(J/ψ)\,τ\,ν_τ$, and $B_c \to D^{(*)}\,τ\,ν_τ$ within the \emph{Relativistic Independent Quark (RIQ) Model}, emphasizing a quark potential model based analysis of these transitions. We outline the essential elements of the model, incorporating corrections from residual interactions and center-of-mass motion, and perform a comprehensive study of the form factors across the full physical kinematic range of $q^2$. The resulting predictions demonstrate consistency and good agreement with existing theoretical approaches and experimental measurements. Motivated by recent observations of polarization observables at LHCb and Belle, we further evaluate these quantities within our framework and find results compatible with Standard Model (SM) expectations. The predictions presented here serve as theoretical input in decay channels for which Lattice QCD results remain limited, offering guidance for future experimental and Lattice efforts. Thus, semileptonic $B$ decays continue to serve as precise and discerning probes of the fundamental mechanisms governing flavor transitions in the SM.

---

## 133. Sticky eigenstates in systems with sharply-divided phase space

**论文链接:** [http://arxiv.org/abs/2512.05627v1](http://arxiv.org/abs/2512.05627v1)

**作者:** Hua Yan

**发布时间:** 2025-12-05

### GPT解析

### 总结

研究具有明显划分相空间的系统中的混合本征态，使用不同分段线性映射分析其行为特征和标度关系

### 背景

研究具有正则-混沌边界由边缘不稳定周期轨道或准周期轨道形成的系统

### 目的

分类混合本征态并研究动力学隧穿贡献和粘附本征态的标度行为

### 方法

使用重叠指数和熵局域长度对混合本征态进行分类

### 主要发现

动力学隧穿贡献标度为约普朗克常数指数负b除以普朗克常数；粘附本征态在边缘不稳定周期轨道情况下标度为普朗克常数的二分之一次方，在准周期情况下围绕此代数行为振荡；这一行为推广了KAM系统中层次态的预测，标度为普朗克常数的1减去1除以γ次方；对于所研究的映射，γ值为2

### 结论

揭示了非KAM系统中经典粘性的明确量子特征

### 翻译

我们研究具有明显划分相空间的系统中的混合本征态，使用不同的分段线性映射，其正则-混沌边界由边缘不稳定周期轨道或准周期轨道形成。通过重叠指数和熵局域长度，我们对混合本征态进行分类，并表明动力学隧穿的贡献标度为约普朗克常数指数负b除以普朗克常数，其中b大于零与正则区域的相对大小相关。大部分保持与边界粘附的状态，称为粘附本征态，在边缘不稳定周期轨道情况下标度为普朗克常数的二分之一次方，在准周期情况下围绕此代数行为振荡。这一行为推广了KAM系统中层次态的已有预测，其标度为普朗克常数的1减去1除以γ次方，其中γ由相应的经典粘性决定，反映在累积RTDs t的负γ次方的代数衰减中。对于所研究的分段线性映射，γ等于2。这些结果揭示了非KAM系统中经典粘性的明确量子特征。


### 论文摘要

We investigate mixed eigenstates in systems with sharply-divided phase space, using different piecewise-linear maps whose regular-chaotic boundaries are formed by marginally unstable periodic orbits (MUPOs) or by quasi-periodic orbits. With the overlap index and the entropy localization length, we classify mixed eigenstates and show that the contribution from dynamical tunneling scales as $\sim \hbar\, \exp(-b/\hbar)$, with $b>0$ associated with the relative size of the regular region. The dominant fraction of states that remain sticky to the boundaries, referred to as sticky eigenstates, scales as $\hbar^{1/2}$ in the MUPO case and oscillates around this algebraic behavior in the quasi-periodic case. This behavior generalizes established predictions for hierarchical states in KAM systems, which scale as $\hbar^{1 - 1/γ}$, with $γ$ set by the corresponding classical stickiness reflected in the algebraic decay of cumulative RTDs $t^{-γ}$. For the piecewise-linear maps studied here, $γ= 2$. These results reveal a clear quantum signature of classical stickiness in non-KAM systems.

---

## 134. Design-marginal calibration of Gaussian process predictive distributions: Bayesian and conformal approaches

**论文链接:** [http://arxiv.org/abs/2512.05611v1](http://arxiv.org/abs/2512.05611v1)

**作者:** Aurélien Pion, Emmanuel Vazquez

**发布时间:** 2025-12-05

### GPT解析

### 总结

本研究从设计边缘角度探讨了高斯过程在插值设置中的预测分布校准问题，提出了两种新方法cps-gp和bcr-gp，并通过数值实验验证了它们的有效性。

### 背景

高斯过程预测分布在插值设置中的校准问题尚未得到充分研究，特别是在设计边缘角度的考量。

### 目的

研究高斯过程预测分布在插值设置中的校准问题，提出新的校准方法，并评估其性能。

### 方法

1. 形式化了中心区间的μ覆盖率和通过随机概率积分变换的μ概率校准；2. 提出cps-gp方法：使用标准化留一法残差将保形预测系统适应于GP插值；3. 提出bcr-gp方法：保留GP后验均值，用一般正态模型替换高斯残差；4. 基于贝叶斯选择规则控制离散和尾部行为；5. 在基准函数上进行数值实验，比较不同方法的性能。

### 主要发现

1. cps-gp和bcr-gp方法均能有效校准高斯过程预测分布；2. cps-gp产生具有有限样本边缘校准的分步预测分布；3. bcr-gp产生适合顺序设计的平滑预测分布；4. 基于贝叶斯选择规则能有效控制离散和尾部行为。

### 结论

所提出的cps-gp和bcr-gp方法在高斯过程预测分布校准方面表现良好，适用于插值设置，且能产生不同特性的预测分布以适应不同应用需求。

### 翻译

我们研究了从设计边缘角度研究高斯过程在插值设置中的预测分布校准问题。通过条件化数据和在设计测度μ上进行平均，我们形式化了中心区间的μ覆盖率和通过随机概率积分变换的μ概率校准。我们介绍了两种方法。cps-gp使用标准化留一法残差将保形预测系统适应于GP插值，产生具有有限样本边缘校准的分步预测分布。bcr-gp保留GP后验均值，并用拟合到交叉验证标准化残差的一般正态模型替换高斯残差。基于贝叶斯选择规则—基于方差的贝叶斯后验上分位数用于保守预测或基于交叉后验Kolmogorov-Smirnov标准用于概率校准—控制离散和尾部行为，同时产生适合顺序设计的平滑预测分布。在基准函数上的数值实验比较了cps-gp、bcr-gp、GP的Jackknife+和完整保形高斯过程，使用校准指标（覆盖率、Kolmogorov-Smirnov、积分绝对误差）和通过缩放连续排名概率评分的准确性或锐度。


### 论文摘要

We study the calibration of Gaussian process (GP) predictive distributions in the interpolation setting from a design-marginal perspective. Conditioning on the data and averaging over a design measure μ, we formalize μ-coverage for central intervals and μ-probabilistic calibration through randomized probability integral transforms. We introduce two methods. cps-gp adapts conformal predictive systems to GP interpolation using standardized leave-one-out residuals, yielding stepwise predictive distributions with finite-sample marginal calibration. bcr-gp retains the GP posterior mean and replaces the Gaussian residual by a generalized normal model fitted to cross-validated standardized residuals. A Bayesian selection rule-based either on a posterior upper quantile of the variance for conservative prediction or on a cross-posterior Kolmogorov-Smirnov criterion for probabilistic calibration-controls dispersion and tail behavior while producing smooth predictive distributions suitable for sequential design. Numerical experiments on benchmark functions compare cps-gp, bcr-gp, Jackknife+ for GPs, and the full conformal Gaussian process, using calibration metrics (coverage, Kolmogorov-Smirnov, integral absolute error) and accuracy or sharpness through the scaled continuous ranked probability score.

---

## 135. Machine and Deep Learning Regression for Compact Object Equations of State

**论文链接:** [http://arxiv.org/abs/2512.05566v1](http://arxiv.org/abs/2512.05566v1)

**作者:** I. Stergakis, Th. Diakonidis, Ch. C. Moustakidis

**发布时间:** 2025-12-05

**备注:** 15 pages, 14 figures. Any comments are welcome

### GPT解析

### 总结

本研究利用先进的机器学习和深度学习技术分析致密天体的质量-半径关系，旨在重建或推断其底层方程状态，为理解超核密度物质行为提供新见解。

### 背景

核物理的核心挑战之一是确定致密核物质的物理稳健方程状态(EoS)，这直接影响对中子星和夸克星等致密天体内部成分和宏观性质的理解。传统方法主要依赖理论建模，并通过重离子碰撞和多信使天体物理观测进行验证。

### 目的

利用最先进的机器学习和深度学习技术分析致密物体的质量-半径关系，以重建或推断其底层方程状态。

### 方法

基于一个广泛的物理一致、多模态中子星EoS库和相应的夸克星EoS集合，每个都设计为满足已建立的理论和观测约束，并利用这些计算框架的预测能力。

### 主要发现

数据驱动方法为理解超核密度物质行为提供了更深入的见解，并为致密物质EoS的更统一理解做出贡献。

### 结论

数据驱动方法在分析致密天体的质量-半径关系和推断其底层方程状态方面显示出显著潜力，有助于增进对超核密度物质行为的理解。

### 翻译

核物理学中的一个核心开放问题是确定致密核物质的物理稳健方程状态(EoS)，这直接影响了我们对中子星和夸克星等致密天体内部成分和宏观性质的理解。传统努力主要依赖于基于核物理和粒子物理的理论建模，随后通过与重离子碰撞的经验约束以及越来越多的多信使天体物理学观测进行验证。然而，最近的发展引入了互补的分析策略，将理论建模与先进的数据驱动方法相结合。特别是，贝叶斯推断、机器学习和深度学习已成为约束EoS和从复杂观测数据中提取物理见解的强大工具。在本工作中，我们采用最先进的机器学习和深度学习技术来分析致密物体的质量-半径关系，旨在重建或推断其底层方程状态。该分析基于一个广泛的物理一致、多模态中子星EoS库和相应的夸克星EoS集合，每个都构建为满足已建立的理论和观测约束。通过利用这些计算框架的预测能力，我们证明了数据驱动方法为理解超核密度物质行为提供更精细见解的潜力，并为致密物质EoS的更统一理解做出贡献。


### 论文摘要

A central open problem in nuclear physics is the determination of a physically robust equation of state (EoS) for dense nuclear matter, which directly informs our understanding of the internal composition and macroscopic properties of compact objects such as neutron stars and quark stars. Traditional efforts have relied primarily on theoretical modeling grounded in nuclear and particle physics, with subsequent validation against empirical constraints from heavy ion collisions and, increasingly, multimessenger astrophysical observations. Recent developments, however, have introduced complementary analytical strategies that merge theoretical modeling with advanced data driven methodologies. In particular, Bayesian inference, machine learning, and deep learning have emerged as powerful tools for constraining the EoS and extracting physical insight from complex observational datasets. In this work, we employ state of the art machine learning and deep learning techniques to analyze mass radius relations of compact objects with the aim of reconstructing or inferring their underlying equations of state. The analysis is based on an extensive library of physically consistent, multimodal EoSs for neutron stars and a corresponding set for quark stars, each constructed to satisfy established theoretical and observational constraints. By leveraging the predictive capacity of these computational frameworks, we demonstrate the potential of data-driven approaches to provide refined insights into the behavior of matter at supranuclear densities and to contribute to a more unified understanding of the dense matter EoS.

---

## 136. Improving Local Fidelity Through Sampling and Modeling Nonlinearity

**论文链接:** [http://arxiv.org/abs/2512.05556v1](http://arxiv.org/abs/2512.05556v1)

**作者:** Sanjeev Shrestha, Rahul Dubey, Hui Liu

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种使用多元自适应回归样条和非线性边界建模的新方法，用于改进黑盒机器学习模型的解释性，显著提高了局部保真度。

### 背景

随着黑盒机器学习模型日益复杂并被应用于高风险领域，为这些模型的预测提供解释变得至关重要。局部可解释模型无关解释（LIME）是一种广泛使用的解释技术。

### 目的

解决LIME假设局部决策边界为线性而无法捕捉非线性关系的问题，开发一种能够生成高保真度解释的新方法。

### 方法

使用多元自适应回归样条（MARS）对非线性局部边界进行建模，有效捕捉参考模型的底层行为；利用N-ball采样技术直接从期望分布中采样，而不是像LIME那样重新加权样本。

### 主要发现

在三个UCI数据集上对不同分类器和不同核宽度的实验表明，该方法相比基线能产生更可信的解释，平均减少37%的均方根误差，显著提高了局部保真度。

### 结论

通过结合MARS建模非线性边界和N-ball采样技术，所提出的方法有效改进了解释的局部保真度和可信度。

### 翻译

随着黑盒机器学习模型日益复杂及其在高风险领域的应用，为其预测提供解释变得至关重要。局部可解释模型无关解释（LIME）是一种广泛使用的技术，它通过在预测实例周围局部学习一个可解释模型来解释任何分类器的预测。然而，它假设局部决策边界是线性的，无法捕捉非线性关系，导致不正确的解释。在本文中，我们提出了一种能生成高保真度解释的新方法。使用多元自适应回归样条（MARS）对非线性局部边界进行建模，有效捕捉参考模型的底层行为，从而提高了解释的局部保真度。此外，我们利用N-ball采样技术，直接从期望分布中采样，而不是像LIME那样重新加权样本，进一步提高了保真度分数。我们在三个UCI数据集上评估了我们的方法，使用了不同的分类器和变化的核宽度。实验结果表明，与基线方法相比，我们的方法能产生更可信的解释，平均减少37%的均方根误差，显著提高了局部保真度。


### 论文摘要

With the increasing complexity of black-box machine learning models and their adoption in high-stakes areas, it is critical to provide explanations for their predictions. Local Interpretable Model-agnostic Explanation (LIME) is a widely used technique that explains the prediction of any classifier by learning an interpretable model locally around the predicted instance. However, it assumes that the local decision boundary is linear and fails to capture the non-linear relationships, leading to incorrect explanations. In this paper, we propose a novel method that can generate high-fidelity explanations. Multivariate adaptive regression splines (MARS) is used to model non-linear local boundaries that effectively captures the underlying behavior of the reference model, thereby enhancing the local fidelity of the explanation. Additionally, we utilize the N-ball sampling technique, which samples directly from the desired distribution instead of reweighting samples as done in LIME, further improving the faithfulness score. We evaluate our method on three UCI datasets across different classifiers and varying kernel widths. Experimental results show that our method yields more faithful explanations compared to baselines, achieving an average reduction of 37% in root mean square error, significantly improving local fidelity.

---

## 137. Deadline-Chasing in Digital Health: Modeling EMR Adoption Dynamics and Regulatory Impact in Indonesian Primary Care

**论文链接:** [http://arxiv.org/abs/2512.05381v1](http://arxiv.org/abs/2512.05381v1)

**作者:** Suryo Satrio, Bukhori Muhammad Aqid

**发布时间:** 2025-12-05

**备注:** Data collected from PT Medigo Teknologi Kesehatan in collaboration with Ministry of Communication and Information Technology

### GPT解析

### 总结

本研究评估了印度尼西亚基层卫生设施中电子病历(EMR)的采用水平和速率，并建立了短期预测模型。研究发现EMR采用呈现稳定增长趋势，注册当月激活迅速，但总体采用率仍较低，预计到2025年6月累计约3,997个诊所采用EMR系统。

### 背景

印度尼西亚在2022年卫生部长第24号法规推动下加速数字医疗保健转型，要求采用电子病历并与SATUSEHAT平台集成。然而，关于基层卫生设施(FKTP)采用EMR的因素、轨迹和速度的实证证据有限。

### 目的

评估EMR系统提供商PT MTK客户网络中EMR的采用水平和速率，并建立短期预测模型。

### 方法

观察性研究，主要变量包括累计注册EMR设施、月度注册流量、当月激活率、当月停用率和每月全国符合条件的FKTP估计数量。分析方法采用描述性分析、逻辑增长建模和ARIMA预测。

### 主要发现

33个月的研究显示，累计注册设施从2个增加到3,533个，当月激活率中位数为0.889(IQR 0.717至0.992)。相比符合条件的设施，最终采用比例为8.9%。ARIMA模型预测到2025年6月累计约3,997个诊所(95% CI 3,697至4,298)，估计承载能力为4.1千个设施。研究发现存在'截止日期追逐'现象，即截止日期前有局部阶梯式上升。

### 结论

EMR系统提供商客户网络中的EMR采用显示出稳定增长，注册当月激活迅速。建议通过将干预措施与截止日期日历对齐来最大化影响。根据预测轨迹，PT MTK在2024年底的FKTP市场份额仍低于10%，但2025年将继续增长。

### 翻译

印度尼西亚数字医疗保健转型正在加速，根据2022年第24号卫生部长法规，该法规要求采用电子病历(EMR)并与SATUSEHAT平台集成。然而，关于基层卫生设施(FKTP)采用的因素、轨迹和速度的实证证据仍然有限。本研究旨在评估主要EMR系统提供商PT MTK客户网络中EMR的采用水平和速率，并建立短期预测模型。这是一项观察性研究，主要变量包括累计注册EMR设施、月度注册流量、当月激活、当月停用以及全国每月符合条件的FKTP估计数量(称为符合条件的设施)。分析使用描述性分析、逻辑增长建模和ARIMA预测。33个月的研究结果显示，累计注册设施从2个增加到3,533个，当月激活率中位数为0.889(IQR 0.717至0.992)。与符合条件的设施相比，最终采用比例为8.9%(39,852个中的3,533个)。ARIMA模型预测到2025年6月累计约3,997个诊所(95% CI 3,697至4,298)。估计的逻辑增长收敛，承载能力为4.1千个设施。研究结果表明，EMR系统提供商客户网络中的EMR采用显示出稳定增长，注册当月激活迅速。尽管累计系列没有偏离长期趋势的重大变化，但截止日期前的局部阶梯式上升表明存在'截止日期追逐'现象，因此应通过将干预措施与截止日期日历对齐来最大化影响。根据这一轨迹，PT MTK在2024年底的FKTP总市场份额仍低于10%，但在2025年继续增长。


### 论文摘要

Indonesia digital healthcare transformation is accelerating under Minister of Health Regulation Number 24 of 2022, which mandates the adoption of Electronic Medical Records EMR and integration with the SATUSEHAT platform. However, empirical evidence regarding the factors, trajectory and speed of adoption in Primary Health Facilities FKTP remains limited. This study aims to evaluate the level and rate of EMR adoption within the customer network of a major EMR system provider PT MTK and model short-term projections. This is an observational study with the main variables being cumulative registered EMR facilities, monthly registration flow, same-month activation, same-month inactivation, and the estimated number of eligible FKTPs nationally monthly known as eligible facilities. The analysis uses descriptive analysis, logistic growth modeling, and ARIMA forecasting. The results of the study over 33 months showed that cumulative registered facilities increased from 2 to 3,533, with a median same-month activation rate of 0.889 IQR 0.717 to 0.992. The proportion of final adoption compared to eligible facilities was 8.9 percent 3,533 of 39,852. The ARIMA model projects a cumulative approximately 3,997 clinics 95 percent CI 3,697 to 4,298 by June 2025. The estimated growth in logistics converges with a carrying capacity of 4.1 thousand facilities. The study findings reveal that EMR adoption within the customer network of EMR system providers is showing steady growth with rapid activation in the month of registration. Although the cumulative series showed no major departures from the long-term trend, localized step-ups around deadlines suggest deadline chasing, so impact should be maximized by aligning interventions to the deadline calendar. Given the trajectory, total market share of FKTP for PT MTK remains less than 10 percent at the end of 2024, but continues to increase in 2025.

---

## 138. Robustness Test for AI Forecasting of Hurricane Florence Using FourCastNetv2 and Random Perturbations of the Initial Condition

**论文链接:** [http://arxiv.org/abs/2512.05323v1](http://arxiv.org/abs/2512.05323v1)

**作者:** Adam Lizerbram, Shane Stevenson, Iman Khadir, Matthew Tu, Samuel S. P. Shen

**发布时间:** 2025-12-04

**备注:** 26 pages, 12 figures

### GPT解析

### 总结

这项研究测试了NVIDIA的FourCastNetv2人工智能天气预报模型对输入噪声的敏感性和鲁棒性，特别关注飓风预测的可靠性。研究通过注入高斯噪声和使用完全随机初始条件两种方法评估模型表现，发现模型在低至中等噪声下能准确保留飓风特征，但在所有噪声水平下持续低估风暴强度。

### 背景

理解天气预报模型对输入噪声或不同不确定性的鲁棒性对于评估其输出可靠性非常重要，特别是对于飓风等极端天气事件。

### 目的

测试NVIDIA的FourCastNetv2人工智能天气预报模型对输入噪声和不确定性的敏感性和鲁棒性，评估其在预测飓风轨迹和强度方面的表现。

### 方法

进行了两项实验：1) 从ERA5数据集中获取飓风佛罗伦萨初始条件，注入不同量高斯噪声，观察对预测轨迹和强度的影响；2) 使用完全随机初始条件启动模型，观察其响应。

### 主要发现

1) FCNv2在低至中等噪声下准确保留飓风特征；2) 高噪声下仍保持总体轨迹和结构，但位置准确性下降；3) 所有噪声水平下持续低估风暴强度和持续性；4) 使用随机初始条件时，模型在几步后生成平滑一致的预测。

### 结论

FCNv2模型对初始条件中的噪声表现出良好鲁棒性，能在各种噪声水平下生成合理飓风预测，但存在系统性低估强度问题。该方法简单且可移植到其他AI天气模型中。

### 翻译

理解天气预报模型对输入噪声或不同不确定性的鲁棒性对于评估其输出可靠性非常重要，特别是对于飓风等极端天气事件。在本文中，我们测试了人工智能天气预报模型NVIDIA的FourCastNetv2的敏感性和鲁棒性。我们进行了两项实验，旨在评估模型在不同噪声水平注入到模型初始条件下的输出表现。首先，我们从欧洲中期天气预报中心再分析v5数据集中获取飓风佛罗伦萨的初始条件，用不同量的高斯噪声进行扰动，并检查对预测轨迹和预报风暴强度的影响。其次，我们使用完全随机的初始条件启动FCNv2，观察模型如何对无意义输入做出响应。我们的结果表明，FCNv2在低至中等噪声注入下能准确保留飓风特征。即使在噪声水平高的情况下，模型仍能保持总体风暴轨迹和结构，尽管位置准确性开始下降。在所有注入噪声水平下，FCNv2持续低估风暴强度和持续性。使用完全随机的初始条件时，模型在几步时间后生成平滑且一致的预测，这表明模型倾向于稳定、平滑输出的特性。我们的方法简单且可移植到其他数据驱动的人工智能天气预报模型中。


### 论文摘要

Understanding the robustness of a weather forecasting model with respect to input noise or different uncertainties is important in assessing its output reliability, particularly for extreme weather events like hurricanes. In this paper, we test sensitivity and robustness of an artificial intelligence (AI) weather forecasting model: NVIDIAs FourCastNetv2 (FCNv2). We conduct two experiments designed to assess model output under different levels of injected noise in the models initial condition. First, we perturb the initial condition of Hurricane Florence from the European Centre for Medium-Range Weather Forecasts (ECMWF) Reanalysis v5 (ERA5) dataset (September 13-16, 2018) with varying amounts of Gaussian noise and examine the impact on predicted trajectories and forecasted storm intensity. Second, we start FCNv2 with fully random initial conditions and observe how the model responds to nonsensical inputs. Our results indicate that FCNv2 accurately preserves hurricane features under low to moderate noise injection. Even under high levels of noise, the model maintains the general storm trajectory and structure, although positional accuracy begins to degrade. FCNv2 consistently underestimates storm intensity and persistence across all levels of injected noise. With full random initial conditions, the model generates smooth and cohesive forecasts after a few timesteps, implying the models tendency towards stable, smoothed outputs. Our approach is simple and portable to other data-driven AI weather forecasting models.

---

## 139. XR-DT: Extended Reality-Enhanced Digital Twin for Agentic Mobile Robots

**论文链接:** [http://arxiv.org/abs/2512.05270v1](http://arxiv.org/abs/2512.05270v1)

**作者:** Tianyi Wang, Jiseop Byeon, Ahmad Yehia, Huihai Wang, Yiming Xu, Tianyi Zeng, Ziran Wang, Junfeng Jiao, Christian Claudel

**发布时间:** 2025-12-04

**备注:** 10 pages, 5 figures

### GPT解析

### 总结

本文提出XR-DT框架，一种扩展现实增强的数字孪生系统，用于解决移动机器人在共享工作空间中与人类交互的安全、效率和可解释性问题。该框架通过连接物理和虚拟空间，实现人机双向理解。

### 背景

随着移动机器人在共享工作空间中与人类协同操作日益增多，确保安全、高效和可解释的人机交互成为紧迫挑战。尽管人类行为预测研究取得进展，但人类如何感知、解释和信任机器人的推理研究有限，阻碍了机器人在安全关键和社会嵌入环境中的部署。

### 目的

开发XR-DT框架，通过扩展现实增强的数字孪生技术，连接物理和虚拟空间，实现人类和机器人之间的双向理解，促进可解释、可信和自适应的人机交互。

### 方法

XR-DT采用分层架构，集成虚拟现实、增强现实和混合现实层，融合实时传感器数据、Unity引擎模拟环境和可穿戴AR设备捕获的人类反馈。设计具有统一扩散策略的代理移动机器人系统用于上下文感知任务适应，提出思维链提示机制使多模态大语言模型能够推理人类指令和环境上下文，并利用基于AutoGen的多代理协调层增强动态任务中的鲁棒性和协作能力。

### 主要发现

初始实验结果表明实现了准确的人类和机器人轨迹预测，验证了XR-DT框架在人机交互任务中的有效性。

### 结论

通过将人类意图、环境动态和机器人认知嵌入XR-DT框架，该系统能够实现可解释、可信和自适应的人机交互，为移动机器人在人类共享环境中的安全部署提供了新途径。

### 翻译

随着移动机器人在共享工作空间中与人类一起操作日益增多，确保安全、高效和可解释的人机交互已成为一个紧迫的挑战。虽然在人类行为预测方面取得了重大进展，但对人类如何感知、解释和信任机器人的推理关注有限，阻碍了机器人在安全关键和社会嵌入环境中的部署。本文提出了XR-DT，这是一种用于代理移动机器人的扩展现实增强数字孪生框架，通过连接物理和虚拟空间实现人机之间的双向理解。我们的分层XR-DT架构集成了虚拟现实、增强现实和混合现实层，融合实时传感器数据、Unity游戏引擎中的模拟环境以及通过可穿戴AR设备捕获的人类反馈。在此框架内，我们设计了一个具有统一扩散策略的代理移动机器人系统，用于上下文感知的任务适应。我们进一步提出了一种思维链提示机制，使多模态大语言模型能够对人类指令和环境上下文进行推理，同时利用基于AutoGen的多代理协调层来增强动态任务中的鲁棒性和协作能力。初步实验结果表明实现了准确的人类和机器人轨迹预测，验证了XR-DT框架在人机交互任务中的有效性。通过将人类意图、环境动态和机器人认知嵌入XR-DT框架，我们的系统实现了可解释、可信和自适应的人机交互。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决移动机器人在与人类共享工作空间时的安全、高效和可解释的人机交互问题。现有研究主要关注人类行为预测，而忽视了人类如何感知、解释和信任机器人的推理过程，这限制了机器人在安全关键环境和社会嵌入环境中的部署。这个问题很重要，因为随着机器人越来越多地与人类协作，确保人类理解并信任机器人的决策对于广泛应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到数字孪生技术可以作为物理世界在数字世界的反映，但发现现有研究大多在纯VR环境中进行，无法实现实时监控。他们注意到AR技术的发展促进了人机交互方法的改进，而AI代理的最新进展使代理能够实现上下文感知和自主性。作者借鉴了数字孪生技术、多模态大语言模型、扩散模型和AutoGen框架等现有工作，但创新性地将它们与XR技术结合，创建了XR-DT框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个扩展现实增强的数字孪生框架，连接物理和虚拟空间，实现人类和机器人之间的双向理解。整体流程包括：1) VR增强数字孪生层作为模拟和预测推理空间；2) AR增强数字孪生层作为物理世界接口，将数字信息嵌入真实环境；3) MR增强数字孪生层整合前两层，实现跨世界交互。系统使用链式思维提示策略分析人类指令，统一的扩散策略让机器人适应任务，AutoGen框架实现多智能体协作，最终通过AR设备将预测结果反馈给人类。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 分层XR增强数字孪生架构，整合VR/AR/MR三层；2) 代理移动机器人系统，结合统一扩散策略和链式思维提示；3) 基于AutoGen的多智能体协调层。相比之前工作，XR-DT实现了人类与机器人之间的双向理解而非单向预测，整合了XR技术而非仅限于VR，实现了实时同步而非离线处理，支持多模态数据融合，并能适应动态环境和多用户、多机器人协作场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'XR-DT通过整合扩展现实技术、数字孪生框架和代理人工智能，实现了人类与移动机器人之间的双向理解和协作，提高了人机交互的安全性、可解释性和适应性。'}


### 论文摘要

As mobile robots increasingly operate alongside humans in shared workspaces, ensuring safe, efficient, and interpretable Human-Robot Interaction (HRI) has become a pressing challenge. While substantial progress has been devoted to human behavior prediction, limited attention has been paid to how humans perceive, interpret, and trust robots' inferences, impeding deployment in safety-critical and socially embedded environments. This paper presents XR-DT, an eXtended Reality-enhanced Digital Twin framework for agentic mobile robots, that bridges physical and virtual spaces to enable bi-directional understanding between humans and robots. Our hierarchical XR-DT architecture integrates virtual-, augmented-, and mixed-reality layers, fusing real-time sensor data, simulated environments in the Unity game engine, and human feedback captured through wearable AR devices. Within this framework, we design an agentic mobile robot system with a unified diffusion policy for context-aware task adaptation. We further propose a chain-of-thought prompting mechanism that allows multimodal large language models to reason over human instructions and environmental context, while leveraging an AutoGen-based multi-agent coordination layer to enhance robustness and collaboration in dynamic tasks. Initial experimental results demonstrate accurate human and robot trajectory prediction, validating the XR-DT framework's effectiveness in HRI tasks. By embedding human intention, environmental dynamics, and robot cognition into the XR-DT framework, our system enables interpretable, trustworthy, and adaptive HRI.

---

## 140. Variational Quantum Rainbow Deep Q-Network for Optimizing Resource Allocation Problem

**论文链接:** [http://arxiv.org/abs/2512.05946v1](http://arxiv.org/abs/2512.05946v1)

**作者:** Truong Thanh Hung Nguyen, Truong Thinh Nguyen, Hung Cao

**发布时间:** 2025-12-05

**DOI:** 10.1145/3748522.3779769

**备注:** Quantum Software Engineering Practices at The 41st ACM/SIGAPP Symposium On Applied Computing (SAC 2026)

### GPT解析

### 总结

研究提出了一种结合量子计算与深度强化学习的变分量子Rainbow DQN方法，用于解决资源分配问题。

### 背景

资源分配问题因组合复杂性而属于NP难问题。虽然深度强化学习方法如Rainbow DQN通过优先回放和分布头提高了可扩展性，但经典函数近似器限制了其表示能力。

### 目的

引入变分量子Rainbow DQN（VQR-DQN），整合环形拓扑变分量子电路与Rainbow DQN，利用量子叠加和纠缠特性增强资源分配能力。

### 方法

将人力资源分配问题构建为基于人员能力、事件时间和转换时间的马尔可夫决策过程，具有组合动作空间。

### 主要发现

在四个HRAP基准测试中，VQR-DQN相比随机基线实现了26.8%的标准化完工时间减少，比Double DQN和经典Rainbow DQN性能提高4.9-13.4%，这些增益与电路表达能力、纠缠和策略质量之间的理论联系一致。

### 结论

证明了增强型量子深度强化学习在解决大规模资源分配问题中的潜力，实现可在提供的GitHub链接获取。

### 翻译

资源分配由于组合复杂性仍然是NP难问题。虽然深度强化学习方法，如Rainbow深度Q网络，通过优先回放和分布头提高了可扩展性，但经典函数近似器限制了它们的表示能力。我们引入了变分量子Rainbow DQN（VQR-DQN），它将环形拓扑变分量子电路与Rainbow DQN相结合，利用量子叠加和纠缠。我们将人力资源分配问题（HRAP）构建为基于人员能力、事件时间和转换时间的具有组合动作空间的马尔可夫决策过程（MDP）。在四个HRAP基准测试中，VQR-DQN相比随机基线实现了26.8%的标准化完工时间减少，并比Double DQN和经典Rainbow DQN性能高出4.9-13.4%。这些增益与电路表达能力、纠缠和策略质量之间的理论联系一致，证明了增强型量子深度强化学习在大规模资源分配中的潜力。我们的实现可在https://github.com/Analytics-Everywhere-Lab/qtrl/获取。


### 论文摘要

Resource allocation remains NP-hard due to combinatorial complexity. While deep reinforcement learning (DRL) methods, such as the Rainbow Deep Q-Network (DQN), improve scalability through prioritized replay and distributional heads, classical function approximators limit their representational power. We introduce Variational Quantum Rainbow DQN (VQR-DQN), which integrates ring-topology variational quantum circuits with Rainbow DQN to leverage quantum superposition and entanglement. We frame the human resource allocation problem (HRAP) as a Markov decision process (MDP) with combinatorial action spaces based on officer capabilities, event schedules, and transition times. On four HRAP benchmarks, VQR-DQN achieves 26.8% normalized makespan reduction versus random baselines and outperforms Double DQN and classical Rainbow DQN by 4.9-13.4%. These gains align with theoretical connections between circuit expressibility, entanglement, and policy quality, demonstrating the potential of quantum-enhanced DRL for large-scale resource allocation. Our implementation is available at: https://github.com/Analytics-Everywhere-Lab/qtrl/.

---

## 141. NICE: Neural Implicit Craniofacial Model for Orthognathic Surgery Prediction

**论文链接:** [http://arxiv.org/abs/2512.05920v1](http://arxiv.org/abs/2512.05920v1)

**作者:** Jiawen Yang, Yihui Cao, Xuanyu Tian, Yuyao Zhang, Hongjiang Wei

**发布时间:** 2025-12-05

### GPT解析

### 总结

本研究提出了神经颅面隐式模型(NICE)，用于正颌外科手术后的面部外观预测。该模型通过结合形状模块和手术模块，利用隐式神经表示和区域特定的解码器，有效解决了骨骼移动与面部软组织之间复杂非线性相互作用的建模难题，显著提高了预测准确性，特别是在嘴唇和下巴等关键区域。

### 背景

正颌外科手术是矫正牙面骨骼畸形的重要干预手段，可改善咬合功能和面部美观。然而，由于骨骼移动与面部软组织之间存在复杂的非线性相互作用，术后面部外观的准确预测仍然具有挑战性。现有的生物力学模型、参数化模型和深度学习方法要么缺乏计算效率，要么无法完全捕捉这些复杂的相互作用。

### 目的

解决现有方法的局限性，开发一种能够准确预测正颌外科手术结果的模型，同时保持计算效率和捕捉复杂非线性相互作用的能力。

### 方法

提出神经颅面隐式模型(NICE)，采用隐式神经表示进行精确的解剖重建和手术结果预测。NICE包含形状模块(使用区域特定的隐式符号距离函数解码器重建面部表面、上颌和下颌)和手术模块(使用区域特定的变形解码器)。变形解码器由共享的外科潜在代码驱动，建模面部表面对骨骼移动的复杂非线性生物力学响应，并融入解剖先验知识，输出逐点位移场以精确建模手术结果。

### 主要发现

大量实验表明NICE优于当前最先进的方法，特别是在嘴唇和下巴等关键面部区域显著提高了预测准确性，同时稳健地保持了解剖完整性。

### 结论

这项工作为正颌手术提供了临床可行的工具，可以增强手术规划和患者咨询，有助于医生和患者更好地理解手术预期结果。

### 翻译

正颌外科手术是矫正牙面骨骼畸形的重要干预手段，旨在改善咬合功能和面部美观。由于骨骼移动与面部软组织之间存在复杂的非线性相互作用，术后面部外观的准确预测仍然具有挑战性。现有的生物力学模型、参数化模型和深度学习方法要么缺乏计算效率，要么无法完全捕捉这些复杂的相互作用。为解决这些局限性，我们提出了神经颅面隐式模型(NICE)，该模型采用隐式神经表示进行精确的解剖重建和手术结果预测。NICE包含一个形状模块，该模块使用区域特定的隐式符号距离函数(SDF)解码器来重建面部表面、上颌和下颌；以及一个手术模块，该模块使用区域特定的变形解码器。这些变形解码器由共享的外科潜在代码驱动，有效建模面部表面对骨骼移动的复杂非线性生物力学响应，并融入解剖先验知识。变形解码器输出逐点位移场，能够精确建模手术结果。大量实验表明，NICE优于当前最先进的方法，特别是在嘴唇和下巴等关键面部区域显著提高了预测准确性，同时稳健地保持了解剖完整性。这项工作为正颌手术提供了临床可行的工具，可增强手术规划和患者咨询。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决正颌手术中准确预测术后面部外观的问题。这个问题很重要，因为准确的预测能帮助医生和患者沟通、优化手术策略、提高患者满意度，并提供更好的手术规划，避免不必要的并发症。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有方法的局限性：传统生物力学模型精确但计算量大；参数化模型效率高但忽略骨骼结构；深度学习方法效率高但缺乏几何一致性。基于这些分析，作者借鉴了隐式神经表示和区域特定设计，结合了解剖先验知识，设计出NICE方法，解决了现有方法的不足。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用隐式神经表示和区域特定解码器，结合共享的手术潜在代码来建模复杂的非线性生物力学响应。流程包括：1)形状模块使用区域特定SDF解码器重建面部和骨骼结构；2)手术模块使用变形解码器预测手术效果；3)结合形状和手术模块，从术前CT数据预测术后面部外观。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)区域特定的隐式SDF解码器实现高保真重建；2)共享手术潜在代码驱动区域特定变形解码器；3)结合解剖先验知识确保解剖一致性。相比之前工作，NICE不需要患者特定网格化、能捕捉非线性关系、保持几何连续性，避免了骨骼穿透皮肤的问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'NICE通过结合区域特定的隐式神经表示和共享的手术潜在代码，实现了正颌手术术后面部外观的高保真预测，同时保持了解剖一致性和计算效率。'}


### 论文摘要

Orthognathic surgery is a crucial intervention for correcting dentofacial skeletal deformities to enhance occlusal functionality and facial aesthetics. Accurate postoperative facial appearance prediction remains challenging due to the complex nonlinear interactions between skeletal movements and facial soft tissue. Existing biomechanical, parametric models and deep-learning approaches either lack computational efficiency or fail to fully capture these intricate interactions. To address these limitations, we propose Neural Implicit Craniofacial Model (NICE) which employs implicit neural representations for accurate anatomical reconstruction and surgical outcome prediction. NICE comprises a shape module, which employs region-specific implicit Signed Distance Function (SDF) decoders to reconstruct the facial surface, maxilla, and mandible, and a surgery module, which employs region-specific deformation decoders. These deformation decoders are driven by a shared surgical latent code to effectively model the complex, nonlinear biomechanical response of the facial surface to skeletal movements, incorporating anatomical prior knowledge. The deformation decoders output point-wise displacement fields, enabling precise modeling of surgical outcomes. Extensive experiments demonstrate that NICE outperforms current state-of-the-art methods, notably improving prediction accuracy in critical facial regions such as lips and chin, while robustly preserving anatomical integrity. This work provides a clinically viable tool for enhanced surgical planning and patient consultation in orthognathic procedures.

---

## 142. Learning the Cosmic Web: Graph-based Classification of Simulated Galaxies by their Dark Matter Environments

**论文链接:** [http://arxiv.org/abs/2512.05909v1](http://arxiv.org/abs/2512.05909v1)

**作者:** Dakshesh Kololgi, Krishna Naidoo, Amelie Saintonge, Ofer Lahav

**发布时间:** 2025-12-05

**备注:** 15 pages, 7 figures, 9 tables, submitted to Royal Astronomical Society Techniques and Instruments

### GPT解析

### 总结

提出了一种基于图的机器学习分类器，用于识别星系的暗物质宇宙网环境

### 背景

大型星系调查提供了星系属性如何被大尺度结构塑造的全面统计视图，但这需要稳健的星系宇宙网环境分类

### 目的

开发一种方法来识别星系的暗物质宇宙网环境，以便研究星系属性如何受大尺度结构影响

### 方法

使用三阶段基于模拟的框架：1)使用T-Web分类将星系分配到空洞、墙、纤维或团簇环境；2)构建星系分布的Delaunay三角剖分总结局部几何结构；3)训练图注意力网络预测宇宙网环境

### 主要发现

对于恒星质量大于10^9太阳质量的星系，GAT+模型准确率达到85%，优于图不可知的多层感知机和图卷积网络

### 结论

基于图的星系位置表示为推断暗物质环境提供了一种强大且物理上有意义的方法

### 翻译

我们提出了一种新颖的基于图的机器学习分类器，用于识别星系的暗物质宇宙网环境。大型星系调查提供了星系属性如何被大尺度结构塑造的全面统计视图，但这需要稳健的星系宇宙网环境分类。使用恒星质量选择的IllustrisTNG-300星系，我们应用三阶段基于模拟的框架将星系与总（主要是暗）底层物质分布联系起来。我们应用以下三个步骤：首先，我们使用底层物质分布的T-Web分类将模拟星系的位置分配到空洞、墙、纤维或团簇环境。其次，我们构建星系分布的Delaunay三角剖分，用十个图指标总结每个星系的局部几何结构。第三，我们在每个星系的图指标上训练图注意力网络来预测其宇宙网环境。对于恒星质量大于10^9太阳质量的星系，我们的GAT+模型实现了85%的准确率，优于图不可知的多层感知机和图卷积网络。我们的结果表明，星系位置的基于图的表示为推断暗物质环境提供了一种强大且物理上有意义的方法。我们计划将这种基于模拟的图建模应用于研究DESI调查中观测到的星系属性如何受到其暗物质环境的影响。


### 论文摘要

We present a novel graph-based machine learning classifier for identifying the dark matter cosmic web environments of galaxies. Large galaxy surveys offer comprehensive statistical views of how galaxy properties are shaped by large-scale structure, but this requires robust classifications of galaxies' cosmic web environments. Using stellar mass-selected IllustrisTNG-300 galaxies, we apply a three-stage, simulation-based framework to link galaxies to the total (mainly dark) underlying matter distribution. Here, we apply the following three steps: First, we assign the positions of simulated galaxies to a void, wall, filament, or cluster environment using the T-Web classification of the underlying matter distribution. Second, we construct a Delaunay triangulation of the galaxy distribution to summarise the local geometric structure with ten graph metrics for each galaxy. Third, we train a graph attention network (GAT) on each galaxy's graph metrics to predict its cosmic web environment. For galaxies with stellar mass $\mathrm{>10^9 M_{\odot}}$, our GAT+ model achieves an accuracy of $85\,\%$, outperforming graph-agnostic multilayer perceptrons and graph convolutional networks. Our results demonstrate that graph-based representations of galaxy positions provide a powerful and physically meaningful way to infer dark matter environments. We plan to apply this simulation-based graph modelling to investigate how the properties of observed galaxies from the Dark Energy Spectroscopic Instrument (DESI) survey are influenced by their dark matter environments.

---

## 143. Toward Efficient and Robust Behavior Models for Multi-Agent Driving Simulation

**论文链接:** [http://arxiv.org/abs/2512.05812v1](http://arxiv.org/abs/2512.05812v1)

**作者:** Fabian Konstantinidis, Moritz Sackmann, Ulrich Hofmann, Christoph Stiller

**发布时间:** 2025-12-05

**备注:** This work has been submitted to the IEEE for possible publication

### GPT解析

### 总结

该研究提出了一种可扩展的多智能体驾驶模拟方法，通过优化交通参与者的行为模型，实现了既真实又计算高效的模拟效果。

### 背景

可扩展的多智能体驾驶模拟需要既真实又计算高效的行为模型。

### 目的

优化控制单个交通参与者的行为模型，以提高效率并保持真实性。

### 方法

采用基于实例的场景表示，每个交通参与者和地图元素都在其局部坐标系中建模；使用查询中心对称上下文编码器与局部帧之间的相对位置编码来建模交互；采用对抗逆向强化学习学习行为模型；提出自适应奖励变换以在训练过程中自动平衡鲁棒性和真实性。

### 主要发现

该方法能够随着令牌数量高效扩展，显著减少训练和推理时间，同时在位置准确性和鲁棒性方面优于多个基于智能体的基线。

### 结论

所提出的方法在保持真实性的同时提高了计算效率，适用于大规模多智能体驾驶模拟。

### 翻译

可扩展的多智能体驾驶模拟需要既真实又计算高效的行为模型。我们通过优化控制单个交通参与者的行为模型来解决这个问题。为了提高效率，我们采用基于实例的场景表示，其中每个交通参与者和地图元素都在其自身的局部坐标系中建模。这种设计能够实现高效、视角不变的场景编码，并允许静态地图令牌在模拟步骤中重复使用。为了建模交互，我们采用查询中心对称上下文编码器，具有局部帧之间的相对位置编码。我们使用对抗逆向强化学习来学习行为模型，并提出了一种自适应奖励变换，在训练过程中自动平衡鲁棒性和真实性。实验表明，我们的方法能够随着令牌数量高效扩展，显著减少训练和推理时间，同时在位置准确性和鲁棒性方面优于多个基于智能体的基线。


### 论文摘要

Scalable multi-agent driving simulation requires behavior models that are both realistic and computationally efficient. We address this by optimizing the behavior model that controls individual traffic participants. To improve efficiency, we adopt an instance-centric scene representation, where each traffic participant and map element is modeled in its own local coordinate frame. This design enables efficient, viewpoint-invariant scene encoding and allows static map tokens to be reused across simulation steps. To model interactions, we employ a query-centric symmetric context encoder with relative positional encodings between local frames. We use Adversarial Inverse Reinforcement Learning to learn the behavior model and propose an adaptive reward transformation that automatically balances robustness and realism during training. Experiments demonstrate that our approach scales efficiently with the number of tokens, significantly reducing training and inference times, while outperforming several agent-centric baselines in terms of positional accuracy and robustness.

---

## 144. SCoNE: Spherical Consistent Neighborhoods Ensemble for Effective and Efficient Multi-View Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2512.05540v1](http://arxiv.org/abs/2512.05540v1)

**作者:** Yang Xu, Hang Zhang, Yixiao Ma, Ye Zhu, Kai Ming Ting

**发布时间:** 2025-12-05

### GPT解析

### 总结

SCoNE是一种新颖的多视图异常检测方法，通过数据依赖性邻域表示解决了现有方法在一致邻域捕获和计算效率方面的问题

### 背景

多视图异常检测的核心问题是在所有视图中一致地表示正常实例的局部邻域

### 目的

解决现有方法无法很好地捕获一致邻域以及计算成本高的问题

### 方法

提出SCoNE方法，直接用多视图实例表示一致邻域，利用数据依赖性在稀疏区域产生大邻域，在密集区域产生小邻域

### 主要发现

数据依赖性使不同视图中的局部邻域能够很好地表示为一致邻域，无需学习，时间复杂度从O(N²)降低到O(N)

### 结论

SCoNE在检测精度上具有优势，在大数据集上运行速度比现有方法快几个数量级

### 翻译

多视图异常检测的核心问题是在所有视图中一致地表示正常实例的局部邻域。现有方法独立考虑每个视图中的局部邻域表示，然后通过学习过程捕获所有视图之间的一致邻域。它们存在两个关键问题：首先，无法保证能够很好地捕获一致邻域，特别是当相同邻域在不同视图中位于不同密度区域时，导致检测精度较低；其次，学习过程具有高计算成本，使其不适用于大数据集。为解决这些问题，我们提出了一种名为SCoNE的新方法，它有两个独特特点：(a)一致邻域直接用多视图实例表示，不需要现有方法中使用的中间表示；(b)邻域具有数据依赖性，导致在稀疏区域产生大邻域，在密集区域产生小邻域。数据依赖性使不同视图中的局部邻域能够很好地表示为一致邻域，无需学习，这带来了较低的时间复杂度。实证评估表明，SCoNE具有更高的检测精度，在大数据集上运行速度比现有方法快几个数量级。


### 论文摘要

The core problem in multi-view anomaly detection is to represent local neighborhoods of normal instances consistently across all views. Recent approaches consider a representation of local neighborhood in each view independently, and then capture the consistent neighbors across all views via a learning process. They suffer from two key issues. First, there is no guarantee that they can capture consistent neighbors well, especially when the same neighbors are in regions of varied densities in different views, resulting in inferior detection accuracy. Second, the learning process has a high computational cost of $\mathcal{O}(N^2)$, rendering them inapplicable for large datasets. To address these issues, we propose a novel method termed \textbf{S}pherical \textbf{C}onsistent \textbf{N}eighborhoods \textbf{E}nsemble (SCoNE). It has two unique features: (a) the consistent neighborhoods are represented with multi-view instances directly, requiring no intermediate representations as used in existing approaches; and (b) the neighborhoods have data-dependent properties, which lead to large neighborhoods in sparse regions and small neighborhoods in dense regions. The data-dependent properties enable local neighborhoods in different views to be represented well as consistent neighborhoods, without learning. This leads to $\mathcal{O}(N)$ time complexity. Empirical evaluations show that SCoNE has superior detection accuracy and runs orders-of-magnitude faster in large datasets than existing approaches.

---

## 145. ParaUni: Enhance Generation in Unified Multimodal Model with Reinforcement-driven Hierarchical Parallel Information Interaction

**论文链接:** [http://arxiv.org/abs/2512.05422v1](http://arxiv.org/abs/2512.05422v1)

**作者:** Jiangtong Tan, Lin Liu, Jie Huanng, Xiaopeng Zhang, Qi Tian, Feng Zhao

**发布时间:** 2025-12-05

### GPT解析

### 总结

论文提出了ParaUni方法，通过并行提取VLM多层次的视觉特征并使用层集成模块(LIM)进行整合，结合逐层动态调整机制(LDAM)，显著提升了统一多模态模型的视觉生成质量。

### 背景

统一的多模态模型通过结合视觉语言模型(VLMs)和扩散模型显著改善了视觉生成，但现有方法由于表示差异巨大，难以充分平衡足够的交互和灵活的实现。

### 目的

解决现有统一多模态模型中交互与实现平衡不足的问题，提高视觉生成质量。

### 方法

1) 从VLM的不同层次并行提取特征以实现全面信息交互；2) 保留灵活的分离架构；3) 将VLM所有层视觉特征并行输入层集成模块(LIM)进行整合；4) 设计逐层动态调整机制(LDAM)利用强化学习对齐层次特性。

### 主要发现

VLM层次包含丰富的分层信息；不同层次对强化学习中不同奖励响应不均等；利用互补的多层特征可显著提高生成质量；在强化学习阶段具有多种奖励提升的潜力。

### 结论

ParaUni通过结合互补的多层特征显著提高了生成质量，并在强化学习阶段显示出多种奖励提升的强大潜力。

### 翻译

统一的多模态模型通过结合视觉语言模型(VLMs)和扩散模型显著改善了视觉生成。然而，由于表示差异巨大，现有方法难以充分平衡足够的交互和灵活的实现。考虑到VLM层次中从低级细节到高级语义的丰富和分层信息，我们提出了ParaUni。它以并行方式从VLM的不同层次提取特征，以实现全面的信息交互，并保留灵活的分离架构以增强统一多模态模型中的生成能力。具体来说，将VLM所有层的视觉特征并行输入层集成模块(LIM)，该模块有效整合细粒度细节和语义抽象，并提供融合表示作为扩散模型的条件。为进一步提高性能，我们揭示这些层次对强化学习(RL)中的不同奖励响应不均等。关键是我们设计了一个逐层动态调整机制(LDAM)，利用强化学习促进多种奖励改进，通过使用RL对齐这些层次的层次特性。大量实验表明，ParaUni利用互补的多层特征显著提高了生成质量，并在强化学习阶段显示出多种奖励提升的强大潜力。代码可在https://github.com/JosephTiTan/ParaUni获取。


### 论文摘要

Unified multimodal models significantly improve visual generation by combining vision-language models (VLMs) with diffusion models. However, existing methods struggle to fully balance sufficient interaction and flexible implementation due to vast representation difference. Considering abundant and hierarchical information in VLM's layers from low-level details to high-level semantics, we propose \textbf{ParaUni}. It extracts features from variants VLM's layers in a \textbf{Para}llel way for comprehensive information interaction and retains a flexible separation architecture to enhance generation in \textbf{Uni}fied multimodal model. Concretely, visual features from all VLM's layers are fed in parallel into a Layer Integration Module (LIM), which efficiently integrates fine-grained details and semantic abstractions and provides the fused representation as a condition to the diffusion model. To further enhance performance, we reveal that these hierarchical layers respond unequally to different rewards in Reinforcement Learning (RL). Crucially, we design a Layer-wise Dynamic Adjustment Mechanism (LDAM) to facilitate multiple reward improvements that aligns the hierarchical properties of these layers using RL. Extensive experiments show ParaUni leverages complementary multi-layer features to substantially improve generation quality and shows strong potential for multiple reward advances during RL stages. Code is available at https://github.com/JosephTiTan/ParaUni.

---

## 146. Sepsis Prediction Using Graph Convolutional Networks over Patient-Feature-Value Triplets

**论文链接:** [http://arxiv.org/abs/2512.05416v1](http://arxiv.org/abs/2512.05416v1)

**作者:** Bozhi Dan, Di Wu, Ji Xu, Xiang Liu, Yiziting Zhu, Xin Shu, Yujie Li, Bin Yi

**发布时间:** 2025-12-05

### GPT解析

### 总结

本文提出了一种名为Triplet-GCN的单分支图卷积模型，用于早期检测脓毒症风险，通过将电子健康记录编码为三元组并在患者-特征图上传播信息，显著优于传统机器学习方法。

### 背景

在重症监护环境中，脓毒症是导致患者疾病和死亡的主要原因，但电子健康记录数据的复杂、稀疏和异质性特点阻碍了其及时检测。

### 目的

开发一种能够有效处理EHR数据并及早预警脓毒症风险的模型，提高检测的准确性和时效性。

### 方法

提出Triplet-GCN模型，将每次就诊表示为患者-特征-值三元组，构建二部EHR图，通过图卷积网络学习患者嵌入并使用轻量级MLP处理。应用特定预处理：数值变量使用中位数插补和标准化，二元特征使用效应编码，稀有分类属性使用低维嵌入的众数插补。使用汇总统计初始化患者节点，保留边上测量值信息。

### 主要发现

在来自三家三级医院的648名患者回顾性队列中(70/30训练-测试分割)，Triplet-GCN在判别和平衡误差指标上一致优于KNN、SVM、XGBoost和随机森林等基线方法，实现了更优的敏感性-特异性权衡和早期预警效用。

### 结论

将EHR编码为三元组并在患者-特征图上传播信息，比特征独立模型产生信息量更大的患者表示，为可部署的脓毒症风险分层提供了一个简单、端到端的解决方案。

### 翻译

在重症监护环境中，脓毒症继续是导致患者疾病和死亡的主要原因；然而，其及时检测受到电子健康记录数据复杂、稀疏和异质性特点的阻碍。我们提出了Triplet-GCN，一种单分支图卷积模型，将每次就诊表示为患者-特征-值三元组，构建二部EHR图，并通过图卷积网络学习患者嵌入，随后使用轻量级多层感知器处理。该流程应用特定类型的预处理——数值变量使用中位数插补和标准化，二元特征使用效应编码，稀有分类属性使用低维嵌入的众数插补——并使用汇总统计量初始化患者节点，同时在边上保留测量值以保留'谁测量了什么以及测量了多少'。在一个来自三家三级医院的回顾性、多中心中国队列(N = 648; 70/30训练-测试分割)中，Triplet-GCN在判别和平衡误差指标上一致地优于强大的表格基线(KNN、SVM、XGBoost、随机森林)，产生了更有利的敏感性-特异性权衡和改进的早期预警整体效用。这些发现表明，将EHR编码为三元组并在患者-特征图上传播信息，比特征独立模型产生信息量更大的患者表示，为可部署的脓毒症风险分层提供了一个简单、端到端的蓝图。


### 论文摘要

In the intensive care setting, sepsis continues to be a major contributor to patient illness and death; however, its timely detection is hindered by the complex, sparse, and heterogeneous nature of electronic health record (EHR) data. We propose Triplet-GCN, a single-branch graph convolutional model that represents each encounter as patient-feature-value triplets, constructs a bipartite EHR graph, and learns patient embeddings via a Graph Convolutional Network (GCN) followed by a lightweight multilayer perceptron (MLP). The pipeline applies type-specific preprocessing -- median imputation and standardization for numeric variables, effect coding for binary features, and mode imputation with low-dimensional embeddings for rare categorical attributes -- and initializes patient nodes with summary statistics, while retaining measurement values on edges to preserve "who measured what and by how much". In a retrospective, multi-center Chinese cohort (N = 648; 70/30 train-test split) drawn from three tertiary hospitals, Triplet-GCN consistently outperforms strong tabular baselines (KNN, SVM, XGBoost, Random Forest) across discrimination and balanced error metrics, yielding a more favorable sensitivity-specificity trade-off and improved overall utility for early warning. These findings indicate that encoding EHR as triplets and propagating information over a patient-feature graph produce more informative patient representations than feature-independent models, offering a simple, end-to-end blueprint for deployable sepsis risk stratification.

---

## 147. LoC-Path: Learning to Compress for Pathology Multimodal Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.05391v1](http://arxiv.org/abs/2512.05391v1)

**作者:** Qingqiao Hu, Weimin Lyu, Meilong Xu, Kehan Qi, Xiaoling Hu, Saumya Gupta, Jiawei Zhou, Chao Chen

**发布时间:** 2025-12-05

**备注:** 20 pages

### GPT解析

### 总结

本研究提出了一种名为LoC-Path的高效多模态大语言模型框架，用于处理全幻灯片图像理解问题。通过减少特征冗余和优化计算方式，该方法在保持与现有最先进模型相当性能的同时，显著降低了计算和内存需求。

### 背景

全幻灯片图像(WSI)理解因其千兆像素尺度和诊断相关区域的极端稀疏性而具有根本性挑战。现有的病理学多模态大语言模型(MLLMs)依赖于处理大量补丁特征的重型幻灯片级编码器，导致计算成本过高。

### 目的

开发一种高效的MLLM框架，替代现有处理WSI的昂贵方法，减少计算和内存消耗，同时保持与现有最先进模型相当的性能。

### 方法

提出LoC-Path框架，包含两个主要部分：1)稀疏令牌合并器(Sparse Token Merger)和MAE预训练重采样器，用于去除局部冗余并压缩全局冗余的瓦片令牌；2)交叉注意力路由适配器(Cross-Attention Routing Adapter)和令牌重要性评分器(Token Importance Scorer)，用于高效集成压缩的视觉表示与语言模型。

### 主要发现

瓦片级别特征表现出强烈的全局和局部冗余，而只有一小部分瓦片真正与任务相关，这为减少计算提供了可能性。

### 结论

LoC-Path框架实现了与现有最先进的整体幻灯片MLLMs相当的性能，同时需要显著更低的计算和内存，为WSI理解提供了一种更高效的解决方案。

### 翻译

全幻灯片图像(WSI)理解由于其千兆像素尺度和诊断相关区域的极端稀疏性而具有根本性挑战。与主要依靠关键区域做出诊断的人类专家不同，现有的病理学幻灯片级多模态大语言模型(MLLMs)依赖于以蛮力方式处理数千个补丁特征的重型幻灯片级编码器，导致过高的计算成本。在这项工作中，我们重新审视了WSI-语言建模范式，并表明瓦片级别特征表现出强烈的全局和局部冗余，而只有一小部分瓦片真正与任务相关。受此观察启发，我们引入了一个名为LoC-Path的高效MLLM框架，用减少冗余的模块替代了昂贵的幻灯片级编码器。我们首先设计了一个稀疏令牌合并器(Sparse Token Merger)和一个MAE预训练重采样器，以去除局部冗余并将全局冗余的瓦片令牌压缩为紧凑的幻灯片级表示集。然后，我们提出了一个交叉注意力路由适配器(Cross-Attention Routing Adapter)和一个令牌重要性评分器(Token Importance Scorer)，以计算高效的方式将压缩的视觉表示与语言模型集成。大量实验证明，我们的方法实现了与现有最先进的整体幻灯片MLLMs相当的性能，同时需要显著更低的计算和内存。


### 论文摘要

Whole Slide Image (WSI) understanding is fundamentally challenging due to its gigapixel scale and the extreme sparsity of diagnostically relevant regions. Unlike human experts who primarily rely on key areas to arrive at a diagnosis, existing slide-level multimodal large language models (MLLMs) for pathology rely on heavy slide-level encoders that process thousands of patch features in a brute-force manner, resulting in excessive computational cost. In this work, we revisit the WSI-language modeling paradigm and show that tile-level features exhibit strong global and local redundancy, whereas only a small subset of tiles are truly task-relevant. Motivated by this observation, we introduce an efficient MLLM framework, called LoC-Path, that replaces the expensive slide-level encoder with redundancy-reducing modules. We first design a Sparse Token Merger (STM) and an MAE-pretrained resampler to remove local redundancy and compress globally redundant tile tokens into a compact slide-level representation set. We then propose a Cross-Attention Routing Adapter (CARA) and a Token Importance Scorer (TIS) to integrate the compressed visual representation with the language model in a computation-efficient manner. Extensive experiments demonstrate that our approach achieves performance comparable to existing state-of-the-art whole-slide MLLMs, while requiring significantly lower computation and memory.

---

## 148. Group Orthogonal Low-Rank Adaptation for RGB-T Tracking

**论文链接:** [http://arxiv.org/abs/2512.05359v1](http://arxiv.org/abs/2512.05359v1)

**作者:** Zekai Shao, Yufan Hu, Jingyuan Liu, Bin Fan, Hongmin Liu

**发布时间:** 2025-12-05

**备注:** 13 pages, 8 figures. Accepted by AAAI 2026. Extended version

### GPT解析

### 总结

论文提出了一种名为Group Orthogonal Low-Rank Adaptation (GOLA)的框架，用于解决RGB-T跟踪中的参数冗余问题，通过组间正交约束策略提升模型性能。

### 背景

参数高效微调已成为RGB-T跟踪领域的一种有前景的范式，通过冻结预训练参数并只微调一小部分参数来实现任务适应，但这些参数形成的秩空间存在显著冗余问题。

### 目的

解决低秩适应在秩空间中的冗余问题，减少不贡献实用信息的秩，使模型能够学习更多样化的知识以应对RGB-T跟踪中的各种挑战。

### 方法

提出GOLA框架，采用秩分解分区策略利用奇异值分解量化秩重要性，冻结关键秩保留预训练先验，将冗余秩分组并施加组间正交约束，强制各组学习互补特征以应对不同挑战。

### 主要发现

GOLA有效减少了参数冗余并增强了特征表示能力，在四个基准数据集上显著优于最先进方法，验证了其在RGB-T跟踪任务中的有效性。

### 结论

通过结构化参数学习和组间正交约束策略，GOLA能够有效利用秩空间，缓解信息冗余，提升模型对RGB-T跟踪任务的适应能力。

### 翻译

参数高效微调已成为RGB-T跟踪领域的一种有前景的范式，通过冻结预训练参数并微调一小部分参数来实现下游任务适应。这些参数形成的秩空间由多个独立秩组成，其表达能力直接影响模型的适应性。然而，定量分析显示低秩适应在秩空间中存在显著冗余，许多秩几乎不贡献实用信息，阻碍了模型学习更多样化知识以应对RGB-T跟踪中的各种挑战。为此，我们提出用于RGB-T跟踪的组正交低秩适应框架，通过结构化参数学习有效利用秩空间。具体而言，我们采用基于奇异值分解的秩分解分区策略量化秩重要性，冻结关键秩保留预训练先验，并将冗余秩分组为组以准备后续正交约束。我们还设计了组间正交约束策略，强制各组学习针对不同挑战的互补特征，从而缓解信息冗余。实验结果表明，GOLA有效减少了参数冗余并增强了特征表示能力，在四个基准数据集上显著优于最先进方法，验证了其在RGB-T跟踪任务中的有效性。


### 论文摘要

Parameter-efficient fine-tuning has emerged as a promising paradigm in RGB-T tracking, enabling downstream task adaptation by freezing pretrained parameters and fine-tuning only a small set of parameters. This set forms a rank space made up of multiple individual ranks, whose expressiveness directly shapes the model's adaptability. However, quantitative analysis reveals low-rank adaptation exhibits significant redundancy in the rank space, with many ranks contributing almost no practical information. This hinders the model's ability to learn more diverse knowledge to address the various challenges in RGB-T tracking. To address this issue, we propose the Group Orthogonal Low-Rank Adaptation (GOLA) framework for RGB-T tracking, which effectively leverages the rank space through structured parameter learning. Specifically, we adopt a rank decomposition partitioning strategy utilizing singular value decomposition to quantify rank importance, freeze crucial ranks to preserve the pretrained priors, and cluster the redundant ranks into groups to prepare for subsequent orthogonal constraints. We further design an inter-group orthogonal constraint strategy. This constraint enforces orthogonality between rank groups, compelling them to learn complementary features that target diverse challenges, thereby alleviating information redundancy. Experimental results demonstrate that GOLA effectively reduces parameter redundancy and enhances feature representation capabilities, significantly outperforming state-of-the-art methods across four benchmark datasets and validating its effectiveness in RGB-T tracking tasks.

---

## 149. Platonic representation of foundation machine learning interatomic potentials

**论文链接:** [http://arxiv.org/abs/2512.05349v1](http://arxiv.org/abs/2512.05349v1)

**作者:** Zhenzhu Li, Aron Walsh

**发布时间:** 2025-12-05

### GPT解析

### 总结

研究发现独立开发的机器学习原子间势(MLIPs)表现出原子环境统计上一致的几何组织，称为柏拉图表示。通过相对于原子锚点投影嵌入，将七种不同架构的MLIPs潜在空间统一到公共度量空间中，实现跨模型最优传输、嵌入算术和表示偏差检测，并能通过几何失真识别物理预测失败。

### 背景

基础机器学习原子间势(MLIPs)在重叠的化学空间上训练，但它们的潜在表示仍然是模型特定的。

### 目的

统一不同MLIPs的潜在空间，实现互操作、可比较和可解释的材料科学基础模型。

### 方法

通过相对于一组原子锚点投影嵌入，将七种MLIPs(包括等变、非等变、保守和非保守架构)的潜在空间统一到一个公共度量空间中，该空间保留了化学周期性和结构不变性。

### 主要发现

1. 独立开发的MLIPs表现出统计上一致的原子环境几何组织(柏拉图表示)；2. 几何失真可以指示物理预测失败，如对称性破缺和不正确的声子色散；3. 不同MLIPs的潜在空间表现出由共享物理和化学约束塑造的一致统计几何。

### 结论

柏拉图表示为材料科学中互操作、可比较和可解释的基础模型提供了实际途径。

### 翻译

基础机器学习原子间势(MLIPs)在重叠的化学空间上训练，但它们的潜在表示仍然是模型特定的。在这里，我们表明独立开发的MLIPs表现出原子环境统计上一致的几何组织，我们称之为柏拉图表示。通过相对于一组原子锚点投影嵌入，我们将七种MLIPs(涵盖等变、非等变、保守和非保守架构)的潜在空间统一到一个公共度量空间中，该空间保留了化学周期性和结构不变性。这种统一框架 enables 直接的跨模型最优传输、可解释的嵌入算术和表示偏差的检测。此外，我们证明了该空间中的几何失真可以表明物理预测失败，包括对称性破缺和不正确的声子色散。我们的结果表明，不同MLIPs的潜在空间表现出由共享物理和化学约束塑造的一致统计几何，表明柏拉图表示为材料科学中互操作、可比较和可解释的基础模型提供了实际途径。


### 论文摘要

Foundation machine learning interatomic potentials (MLIPs) are trained on overlapping chemical spaces, yet their latent representations remain model-specific. Here, we show that independently developed MLIPs exhibit statistically consistent geometric organisation of atomic environments, which we term the Platonic representation. By projecting embeddings relative to a set of atomic anchors, we unify the latent spaces of seven MLIPs (spanning equivariant, non-equivariant, conservative, and non-conservative architectures) into a common metric space that preserves chemical periodicity and structural invariants. This unified framework enables direct cross-model optimal transport, interpretable embedding arithmetic, and the detection of representational biases. Furthermore, we demonstrate that geometric distortions in this space can indicate physical prediction failures, including symmetry breaking and incorrect phonon dispersions. Our results show that the latent spaces of diverse MLIPs present consistent statistical geometry shaped by shared physical and chemical constraints, suggesting that the Platonic representation offers a practical route toward interoperable, comparable, and interpretable foundation models for materials science.

---

## 150. Interaction Tensor Shap

**论文链接:** [http://arxiv.org/abs/2512.05338v1](http://arxiv.org/abs/2512.05338v1)

**作者:** Hiroki Hasegawa, Yukihiko Okada

**发布时间:** 2025-12-05

**备注:** 30 pages

### GPT解析

### 总结

该论文提出了一种名为交互张量SHAP(IT SHAP)的新方法，用于高效计算高维机器学习模型中的高阶特征交互，解决了现有Shapley值方法计算复杂度指数级增长的问题。

### 背景

现代机器学习模型变得越来越深和高维，难以理解单个和组合特征如何影响预测。基于Shapley值的方法虽然提供了特征归因的原则性框架，但现有公式无法有效评估高阶交互。

### 目的

开发一种能够同时保持Shapley Taylor交互指数(STII)的公理化精确性，并避免高阶离散导数固有的指数级计算复杂度的框架。

### 方法

将高阶Shapley交互精确表示为张量网络收缩，在张量列(TT)结构下实现多项式时间和多对数深度计算。引入IT SHAP，将STII重新表述为值张量和权重张量的收缩，并假设权重张量具有多项式TT秩的有限状态TT表示。

### 主要发现

在TT结构的模型和分布张量下，IT SHAP将STII的指数复杂度Θ(4^n)降低到NC2并行时间，为高维模型中的主效应和高阶交互提供了统一、公理化且计算可处理的公式。

### 结论

IT SHAP框架为可扩展的交互感知可解释AI奠定了基础，对大型黑盒模型的交互分析具有重要意义，这些模型的组合结构以前使得交互分析不可行。

### 翻译

机器学习模型变得越来越深和高维，使得难以理解单个和组合特征如何影响它们的预测。虽然基于Shapley值的方法提供了原则性的特征归因，但现有的公式无法有效评估高阶交互：Shapley Taylor交互指数(STII)需要指数级枚举子集，而当前基于张量的方法如边际SHAP张量(MST)仅限于一阶效应。核心问题是，现有框架无法同时保持STII的公理化精确性并避免高阶离散导数固有的指数级计算爆炸。我们证明了高阶Shapley交互可以精确表示为张量网络收缩，在张量列(TT)结构下实现多项式时间和多对数深度计算。我们引入了交互张量SHAP(IT SHAP)，它将STII重新表述为值张量和权重张量的收缩，并假设权重张量具有多项式TT秩的有限状态TT表示。在TT结构的模型和分布张量下，我们证明IT SHAP将STII的指数复杂度Θ(4^n)降低到NC2并行时间。这些结果表明，IT SHAP为高维模型中的主效应和高阶交互提供了统一、公理化且计算可处理的公式。该框架为可扩展的交互感知可解释AI奠定了基础，对大型黑盒模型具有影响，这些模型的组合结构以前使得交互分析不可行。


### 论文摘要

Machine learning models have grown increasingly deep and high dimensional, making it difficult to understand how individual and combined features influence their predictions. While Shapley value based methods provide principled feature attributions, existing formulations cannot tractably evaluate higher order interactions: the Shapley Taylor Interaction Index (STII) requires exponential scale enumeration of subsets, and current tensor based approaches such as the Marginal SHAP Tensor (MST) are restricted to first order effects. The central problem is that no existing framework simultaneously preserves the axiomatic exactness of STII and avoids the exponential computational blow up inherent to high order discrete derivatives. Here we show that high order Shapley interactions can be represented exactly as tensor network contractions, enabling polynomial time and polylog depth computation under Tensor Train (TT) structure. We introduce Interaction Tensor SHAP (IT SHAP), which reformulates STII as the contraction of a Value Tensor and a Weight Tensor, and assume a finite state TT representation of the Weight Tensor with polynomial TT ranks. Under TT structured model and distribution tensors, we show that IT SHAP reduces the exponential complex Theta(4^n) of STII to NC2 parallel time. These results demonstrate that IT SHAP provides a unified, axiomatic, and computationally tractable formulation of main effects and higher order interactions in high dimensional models. This framework establishes a foundation for scalable interaction aware explainable AI, with implications for large black box models whose combinatorial structure has previously rendered interaction analysis infeasible.

---

## 151. IE2Video: Adapting Pretrained Diffusion Models for Event-Based Video Reconstruction

**论文链接:** [http://arxiv.org/abs/2512.05240v1](http://arxiv.org/abs/2512.05240v1)

**作者:** Dmitrii Torbunov, Onur Okuducu, Yi Huang, Odera Dim, Rebecca Coles, Yonggang Cui, Yihui Ren

**发布时间:** 2025-12-04

### GPT解析

### 总结

该研究提出了一种混合捕获范式，结合稀疏RGB关键帧和连续事件流，通过离线重建完整RGB视频来降低连续视频监控中的功耗，同时保持标准视频输出质量。

### 背景

连续视频监控、机器人和可穿戴系统中的传统RGB相机通过固定速率捕获消耗大量能量，而事件相机虽然功耗低，但产生的是异步事件流而非标准RGB视频。

### 目的

提出一种混合捕获范式，减少视频捕获的功耗，同时为下游应用保持标准视频输出，并引入'图像和事件到视频'(IE2Video)任务，从初始帧和事件数据重建RGB视频序列。

### 方法

研究两种架构策略：1) 调整自回归模型(HyperE2VID)用于RGB生成；2) 通过学习编码器和低秩适应将事件表示注入预训练的文本到视频扩散模型(LTX)。

### 主要发现

基于扩散的方法比自回归基线实现33%更好的感知质量(0.283对比0.422 LPIPS)；该方法在三个事件相机数据集和不同序列长度(32-128帧)上展示了跨数据集的强泛化能力。

### 结论

混合捕获范式结合事件相机和RGB关键帧可以有效降低功耗，同时保持视频质量；基于扩散的方法在IE2Video任务上表现优异，具有跨数据集的泛化能力。

### 翻译

监控、机器人和可穿戴系统中的连续视频监控面临一个基本的功耗限制：传统的RGB相机通过固定速率捕获消耗大量能量。事件相机提供稀疏的、由运动驱动的感知，功耗低，但产生的是异步事件流而非RGB视频。我们提出了一种混合捕获范式，记录稀疏的RGB关键帧和连续的事件流，然后离线重建完整的RGB视频——减少捕获功耗，同时为下游应用保持标准视频输出。我们引入了'图像和事件到视频'(IE2Video)任务：从初始帧和后续事件相机数据重建RGB视频序列。我们研究了两种架构策略：调整自回归模型(HyperE2VID)用于RGB生成，以及通过学习编码器和低秩适应将事件表示注入预训练的文本到视频扩散模型(LTX)。我们的实验表明，基于扩散的方法比自回归基线实现33%更好的感知质量(0.283对比0.422 LPIPS)。我们在三个事件相机数据集(BS-ERGB, HS-ERGB远/近)上验证了该方法，展示了不同序列长度(32-128帧)下的跨数据集强泛化能力，在未见过的捕获配置上表现良好。


### 论文摘要

Continuous video monitoring in surveillance, robotics, and wearable systems faces a fundamental power constraint: conventional RGB cameras consume substantial energy through fixed-rate capture. Event cameras offer sparse, motion-driven sensing with low power consumption, but produce asynchronous event streams rather than RGB video. We propose a hybrid capture paradigm that records sparse RGB keyframes alongside continuous event streams, then reconstructs full RGB video offline -- reducing capture power consumption while maintaining standard video output for downstream applications. We introduce the Image and Event to Video (IE2Video) task: reconstructing RGB video sequences from a single initial frame and subsequent event camera data. We investigate two architectural strategies: adapting an autoregressive model (HyperE2VID) for RGB generation, and injecting event representations into a pretrained text-to-video diffusion model (LTX) via learned encoders and low-rank adaptation. Our experiments demonstrate that the diffusion-based approach achieves 33\% better perceptual quality than the autoregressive baseline (0.283 vs 0.422 LPIPS). We validate our approach across three event camera datasets (BS-ERGB, HS-ERGB far/close) at varying sequence lengths (32-128 frames), demonstrating robust cross-dataset generalization with strong performance on unseen capture configurations.

---

## 152. Coefficient of Variation Masking: A Volatility-Aware Strategy for EHR Foundation Models

**论文链接:** [http://arxiv.org/abs/2512.05216v1](http://arxiv.org/abs/2512.05216v1)

**作者:** Rajna Fani, Rafi Al Attrach, David Restrepo, Yugang Jia, Leo Anthony Celi, Peter Schüffler

**发布时间:** 2025-12-04

**备注:** 16 pages, 9 figures, 1 table, 1 algorithm. Accepted at Machine Learning for Health (ML4H) 2025, Proceedings of the Machine Learning Research (PMLR)

### GPT解析

### 总结

该研究提出了一种新的掩码策略CV-Masking，考虑了EHR数据中不同生物标志物的波动性差异，通过自适应调整掩码概率，提高了模型对EHR数据的表示学习效果，增强了下游任务的性能。

### 背景

掩码自编码器(MAEs)越来越多地应用于电子健康记录(EHR)学习通用表示，但现有方法通常采用均匀随机掩码，假设所有特征具有相同可预测性。实际上，实验室测试的波动性存在显著异质性，如钠稳定而乳酸盐波动较大，临床中波动生物标志物常指示急性病理变化，需要更复杂建模。

### 目的

提出一种波动性感知的预训练策略，根据每个特征的内在变异性自适应调整掩码概率，以更好地建模EHR数据中特征的异质性波动。

### 方法

提出变异系数掩码(CV-Masking)策略，结合与临床工作流程一致的仅值掩码目标，与随机和基于方差的掩码策略进行系统性比较。

### 主要发现

在大量实验室测试实验中，CV-M增强了重建能力，改进了下游预测性能，加速了收敛过程，产生了更健壮和具有临床意义的EHR表示。

### 结论

CV-Masking能有效处理EHR数据中特征的波动性异质性，提高模型性能和临床实用性，是一种有效的EHR数据预训练策略。

### 翻译

掩码自编码器(MAEs)越来越多地应用于电子健康记录(EHR)以学习支持多样化临床任务的通用表示。然而，现有方法通常依赖于均匀随机掩码，隐含地假设所有特征都具有相同的可预测性。实际上，实验室测试表现出波动性的显著异质性：一些生物标志物(如钠)保持稳定，而另一些(如乳酸盐)则波动较大，更难建模。从临床角度看，波动的生物标志物通常指示急性病理生理变化，需要更复杂的建模来捕捉其复杂的时间模式。我们提出了一种波动性感知的预训练策略——变异系数掩码(CV-Masking)，它能根据每个特征的内在变异性自适应地调整掩码概率。结合与临床工作流程一致的仅值掩码目标，CV-Masking比随机和基于方差的策略带来系统性改进。在大量实验室测试上的实验表明，CV-Masking增强了重建能力，改进了下游预测性能，加速了收敛过程，产生了更健壮和具有临床意义的EHR表示。


### 论文摘要

Masked autoencoders (MAEs) are increasingly applied to electronic health records (EHR) for learning general-purpose representations that support diverse clinical tasks. However, existing approaches typically rely on uniform random masking, implicitly assuming all features are equally predictable. In reality, laboratory tests exhibit substantial heterogeneity in volatility: some biomarkers (e.g., sodium) remain stable, while others (e.g., lactate) fluctuate considerably and are more difficult to model. Clinically, volatile biomarkers often signal acute pathophysiology and require more sophisticated modeling to capture their complex temporal patterns. We propose a volatility-aware pretraining strategy, Coefficient of Variation Masking (CV-Masking), that adaptively adjusts masking probabilities according to the intrinsic variability of each feature. Combined with a value-only masking objective aligned with clinical workflows, CV-Masking yields systematic improvements over random and variance-based strategies. Experiments on a large panel of laboratory tests show that CV-Masking enhances reconstruction, improves downstream predictive performance, and accelerates convergence, producing more robust and clinically meaningful EHR representations.

---

## 153. Are Bus-Mounted Edge Servers Feasible?

**论文链接:** [http://arxiv.org/abs/2512.05543v1](http://arxiv.org/abs/2512.05543v1)

**作者:** Xuezhi Li, Jiancong He, Ming Xie, Xuyang Chen, Le Chang, Li Jiang, Gui Gui

**发布时间:** 2025-12-05

### GPT解析

### 总结

本研究探讨了基于公交车的车载边缘服务器在车联网中的可行性，通过分析真实数据集和设计优化算法，证明了这种移动边缘服务器能够有效应对动态用户需求，为城市车联网提供了一种可行的边缘计算解决方案。

### 背景

边缘服务器的部署是为车联网提供边缘计算服务的前提。部署在路边单元或基站上的固定边缘服务器能够为道路上的车辆提供基本服务覆盖，但服务器位置和容量固定后，在处理时空用户动态方面效率低下。

### 目的

研究基于真实轨迹的车载边缘服务器的可行性，探索移动服务器为系统增加计算弹性的潜力。

### 方法

首先使用上海公交/出租车/电信数据集调查公交车和基站的覆盖范围；然后建立数学模型并设计贪心启发式算法，以在有限预算下选择有限数量的公交车来最大化需求点覆盖；最后进行基于轨迹的模拟验证算法性能。

### 主要发现

公交车边缘服务器覆盖了很大一部分地理区域和需求点；所提出的算法能够在服务器容量和购买数量等现实约束下有效处理动态用户需求。

### 结论

车载边缘服务器对于城市地区的车联网是可行的、有益的且有价值的。

### 翻译

边缘服务器的部署是为车联网提供边缘计算服务的前提。部署在路边单元或基站上的固定边缘服务器能够为终端用户（即道路上的车辆）提供基本的服务覆盖。然而，服务器位置和容量在部署后是固定的，导致它们在处理时空用户动态方面效率低下。另一方面，像公交车这样的移动服务器有可能为系统增加计算弹性。为此，本文基于真实轨迹研究车载边缘服务器的可行性。首先，我们使用上海公交/出租车/电信数据集调查公交车和基站的覆盖范围，这表明基于公交车的边缘服务器有很大潜力，因为它们覆盖了很大一部分地理区域和需求点。接下来，我们建立了一个数学模型，并设计了一个简单的贪心启发式算法，以选择有限数量的公交车，最大化需求点的覆盖范围，即在有限的购买预算下。我们进行了基于轨迹的模拟来验证所提出的公交车选择算法的性能。结果表明，我们的方法能够在服务器容量和购买数量等现实约束下有效处理动态用户需求。因此，我们声称：车载边缘服务器对于城市地区的车联网是可行的、有益的且有价值的。


### 论文摘要

Placement of edge servers is the prerequisite of provisioning edge computing services for Internet of Vehicles (IoV). Fixed-site edge servers at Road Side Units (RSUs) or base stations are able to offer basic service coverage for end users, i.e., vehicles on road. However, the server locations and capacity are fixed after deployment, rendering their inefficiency in handling spationtemporal user dynamics. Mobile servers such as buses, on the other hand, have the potential of adding computation elasticity to such system. To this end, this paper studies the feasibility of bus-mounted edge servers based on real traces. First, we investigate the coverage of the buses and base stations using the Shanghai bus/taxi/Telecom datasets, which shows a great potential of bus-based edge servers as they cover a great portion of geographic area and demand points. Next, we build a mathematical model and design a simple greedy heuristic algorithm to select a limited number of buses that maximizes the coverage of demand points, i.e., with a limited purchase budget. We perform trace-driven simulations to verify the performance of the proposed bus selection algorithm. The results show that our approach effectively handles the dynamic user demand under realistic constraints such as server capacity and purchase quantity. Thus, we claim: bus-mounted edge servers for vehicular networks in urban areas are feasible, beneficial, and valuable.

---

