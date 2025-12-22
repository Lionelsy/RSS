# 今日论文推荐 - 2025-12-22

共 54 篇论文

---

## 1. Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting

**论文链接:** [http://arxiv.org/abs/2512.17908v1](http://arxiv.org/abs/2512.17908v1)

**作者:** Ananta R. Bhattarai, Helge Rhodin

**发布时间:** 2025-12-19

### GPT解析

### 总结

本研究提出了Re-Depth Anything框架，一种测试时自监督方法，通过融合深度估计模型与大规模2D扩散模型先验知识，解决了基础模型在处理真实世界图像时的性能局限，显著提升了深度估计的准确性和真实感。

### 背景

单目深度估计面临挑战，因为最近的基础模型（如Depth Anything V2）在处理远离训练分布的真实世界图像时表现不佳，存在域差距问题。

### 目的

开发一种能够弥合域差距的方法，提高深度估计模型在真实世界图像上的性能和准确性。

### 方法

提出Re-Depth Anything框架，一种测试时自监督方法，通过融合DA-V2与大规模2D扩散模型先验知识；在输入图像上直接进行无标签精炼，通过重新光照预测的深度图和增强输入；利用形状从明暗恢复（SfS）线索在生成上下文中使用分数蒸馏采样（SDS）替代传统光度重建；采用有针对性优化策略，冻结编码器，只更新中间嵌入并微调解码器。

### 主要发现

Re-Depth Anything在多个基准测试中相比DA-V2在深度准确性和真实感方面取得显著提升；通过增强几何推理开辟了自监督的新途径。

### 结论

Re-Depth Anything框架有效解决了基础模型在处理真实世界图像时的局限性，展示了通过融合不同模型先验知识和创新优化策略提升深度估计性能的有效性。

### 翻译

单目深度估计仍然具有挑战性，因为最近的基础模型，如Depth Anything V2（DA-V2），在处理远离训练分布的真实世界图像时表现不佳。我们引入了Re-Depth Anything，一种测试时自监督框架，通过将DA-V2与大规模2D扩散模型的强大先验知识融合来弥合这种域差距。我们的方法通过重新光照预测的深度图和增强输入，直接在输入图像上进行无标签精炼。这种重新合成方法通过在新的生成上下文中利用从明暗恢复形状（SfS）线索并使用分数蒸馏采样（SDS），替代了传统的光度重建。为了防止优化崩溃，我们的框架采用了一种有针对性的优化策略：我们冻结编码器，只更新中间嵌入，同时微调解码器，而不是直接优化深度或微调整个模型。在多个基准测试中，Re-Depth Anything在深度准确性和真实感方面相比DA-V2取得了显著提升，展示了通过增强几何推理实现自监督的新途径。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决单目深度估计模型（特别是Depth Anything V2）在处理远离训练分布的真实世界图像时表现不佳的问题。这个问题很重要，因为单目深度估计是计算机视觉的基础任务，对3D重建、自动驾驶、机器人导航和虚拟现实等众多应用至关重要。虽然基础模型已取得进展，但在真实场景中仍有提升空间，特别是在处理与训练数据分布差异较大的图像时。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了单目深度估计在真实世界场景中的挑战，特别是测试数据与训练数据分布不匹配的问题。他们借鉴了DreamFusion等利用2D扩散模型作为3D重建先验的工作，以及Shape from Shading原理，但避免了传统方法的局限性。作者创新性地设计了针对性优化策略（冻结编码器，只更新中间嵌入和解码器），并提出了一种新的重照明方法，通过随机光照条件重新照亮预测的深度图，利用2D扩散模型评估结果的真实性来指导优化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用2D扩散模型的强大先验知识，通过自监督方式改进深度估计，通过重照明预测的深度图并使用扩散模型评估结果的真实性来指导深度图的优化，不依赖精确的光度重建，而是通过添加额外的阴影线索来增强几何推理。整体流程：1)接收图像并生成初始深度图；2)从深度图计算法线并使用Blinn-Phong模型合成重照明图像；3)生成文本提示并计算SDS损失；4)联合优化中间特征嵌入和解码器权重；5)多次运行优化并集成结果生成最终深度图。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)新的测试时优化框架，无需额外标记数据；2)单图像重照明模型，可微分连接深度图与输入图像；3)针对性优化方案，联合优化解码器输入嵌入和权重。不同之处：不同于传统光度重建方法，该方法在监督方法之上应用自监督学习；避免精确重建外观，通过重新合成增强输入图像；采用针对性优化策略而非直接优化深度图或微调整个网络；专注于改进现有深度估计模型在真实图像上的表现，而非从文本生成3D内容。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Re-Depth Anything通过自监督重照明和2D扩散模型先验，在测试时优化深度估计模型，显著提高了其在真实世界图像上的深度准确性和真实感。'}


### 论文摘要

Monocular depth estimation remains challenging as recent foundation models, such as Depth Anything V2 (DA-V2), struggle with real-world images that are far from the training distribution. We introduce Re-Depth Anything, a test-time self-supervision framework that bridges this domain gap by fusing DA-V2 with the powerful priors of large-scale 2D diffusion models. Our method performs label-free refinement directly on the input image by re-lighting predicted depth maps and augmenting the input. This re-synthesis method replaces classical photometric reconstruction by leveraging shape from shading (SfS) cues in a new, generative context with Score Distillation Sampling (SDS). To prevent optimization collapse, our framework employs a targeted optimization strategy: rather than optimizing depth directly or fine-tuning the full model, we freeze the encoder and only update intermediate embeddings while also fine-tuning the decoder. Across diverse benchmarks, Re-Depth Anything yields substantial gains in depth accuracy and realism over the DA-V2, showcasing new avenues for self-supervision by augmenting geometric reasoning.

---

## 2. Adversarial Robustness of Vision in Open Foundation Models

**论文链接:** [http://arxiv.org/abs/2512.17902v1](http://arxiv.org/abs/2512.17902v1)

**作者:** Jonathon Fox, William J Buchanan, Pavlos Papadopoulos

**发布时间:** 2025-12-19

**DOI:** 10.1109/ACCESS.2025.3645997

### GPT解析

### 总结

本研究评估了LLaVA-1.5-13B和Meta的Llama 3.2 Vision-8B-2两种视觉语言模型对抗攻击的鲁棒性，发现视觉模态是降低当代开放权重视觉语言模型性能的有效攻击向量，且对抗鲁棒性与标准基准性能不一定直接相关。

### 背景

随着深度学习的发展，理解AI系统识别对象的模型变得越来越困难，攻击者可能通过添加看不见的元素来修改图像，从而混淆AI对实体的识别。

### 目的

研究LLaVA-1.5-13B和Meta的Llama 3.2 Vision-8B-2两种视觉语言模型的对抗鲁棒性。

### 方法

使用未针对的PGD方法针对视觉输入模态进行测试，在视觉问答v2数据集子集上进行实证评估，使用标准的VQA准确率指标量化对抗攻击结果，并比较两种模型的准确率下降情况。

### 主要发现

Llama 3.2 Vision尽管基线准确率较低，但在受到攻击时性能下降较小，特别是在较高扰动水平下；视觉模态是降低当代开放权重视觉语言模型性能的有效攻击向量。

### 结论

对抗鲁棒性不一定与标准基准性能直接相关，可能受到底层架构和训练因素的影响。

### 翻译

随着深度学习的增加，理解AI系统如何识别对象的模型变得越来越困难。因此，攻击者可能通过添加看不见的元素来修改图像，从而混淆AI对实体的识别。本文因此研究了LLaVA-1.5-13B和Meta的Llama 3.2 Vision-8B-2的对抗鲁棒性。这些模型针对视觉输入模态进行了未针对的PGD测试，并在视觉问答v2数据集子集上进行了实证评估。这些对抗攻击的结果然后使用标准的VQA准确率指标进行量化。然后将此评估与LLaVA和Llama 3.2 Vision的准确率下降进行比较。一个关键发现是，Llama 3.2 Vision尽管在此设置中具有较低的基线准确率，但在受到攻击时性能下降较小，特别是在较高的扰动水平下。总体而言，研究结果证实视觉模态是降低当代开放权重视觉语言模型性能的有效攻击向量，包括Meta的Llama 3.2 Vision。此外，研究结果强调对抗鲁棒性不一定与标准基准性能直接相关，可能受到底层架构和训练因素的影响。


### 论文摘要

With the increase in deep learning, it becomes increasingly difficult to understand the model in which AI systems can identify objects. Thus, an adversary could aim to modify an image by adding unseen elements, which will confuse the AI in its recognition of an entity. This paper thus investigates the adversarial robustness of LLaVA-1.5-13B and Meta's Llama 3.2 Vision-8B-2. These are tested for untargeted PGD (Projected Gradient Descent) against the visual input modality, and empirically evaluated on the Visual Question Answering (VQA) v2 dataset subset. The results of these adversarial attacks are then quantified using the standard VQA accuracy metric. This evaluation is then compared with the accuracy degradation (accuracy drop) of LLaVA and Llama 3.2 Vision. A key finding is that Llama 3.2 Vision, despite a lower baseline accuracy in this setup, exhibited a smaller drop in performance under attack compared to LLaVA, particularly at higher perturbation levels. Overall, the findings confirm that the vision modality represents a viable attack vector for degrading the performance of contemporary open-weight VLMs, including Meta's Llama 3.2 Vision. Furthermore, they highlight that adversarial robustness does not necessarily correlate directly with standard benchmark performance and may be influenced by underlying architectural and training factors.

---

## 3. RadarGen: Automotive Radar Point Cloud Generation from Cameras

**论文链接:** [http://arxiv.org/abs/2512.17897v1](http://arxiv.org/abs/2512.17897v1)

**作者:** Tomer Borreda, Fangqiang Ding, Sanja Fidler, Shengyu Huang, Or Litany

**发布时间:** 2025-12-19

**备注:** Project page: https://radargen.github.io/

### GPT解析

### 总结

RadarGen是一种创新的扩散模型，能够从多视角相机图像生成真实的汽车雷达点云，为多模态生成仿真提供了新方向。

### 背景

汽车雷达点云生成是自动驾驶感知系统中的重要挑战，现有方法在生成真实且物理合理的雷达数据方面存在局限。

### 目的

开发一个能够从多视角相机图像合成真实汽车雷达点云的模型，减少生成数据与真实数据之间的差距，促进跨感知模态的统一生成仿真。

### 方法

RadarGen采用扩散模型技术，通过鸟瞰图表示雷达测量并编码空间结构、RCS和多普勒属性；使用轻量级恢复步骤重建点云；并融入从预训练基础模型提取的BEV对齐深度、语义和运动线索，指导生成过程朝向物理合理的雷达模式。

### 主要发现

在大规模驾驶数据上的评估表明，RadarGen能够有效捕捉特征雷达测量分布，显著缩小了与在真实数据上训练的感知模型之间的性能差距。

### 结论

RadarGen代表了跨感知模态统一生成仿真的重要进展，其基于图像的条件化设计使其与现有视觉数据集和仿真框架兼容，为自动驾驶领域提供了可扩展的多模态生成仿真解决方案。

### 翻译

我们提出了RadarGen，一个用于从多视角相机图像合成真实汽车雷达点云的扩散模型。RadarGen通过将雷达测量表示为鸟瞰图形式来将高效的图像-潜在扩散模型适应到雷达领域，该形式编码了空间结构以及雷达散射截面(RCS)和多普勒属性。轻量级恢复步骤从生成的地图中重建点云。为了更好地使生成与视觉场景对齐，RadarGen融合了从预训练基础模型中提取的BEV对齐的深度、语义和运动线索，这些线索将随机生成过程引导向物理上合理的雷达模式。基于图像的条件化使该方法原则上与现有的视觉数据集和仿真框架广泛兼容，为多模态生成仿真提供了可扩展的方向。在大规模驾驶数据上的评估显示，RadarGen捕捉了特征雷达测量分布，并减少了与在真实数据上训练的感知模型之间的差距，标志着跨感知模态统一生成仿真的一步。


### 论文摘要

We present RadarGen, a diffusion model for synthesizing realistic automotive radar point clouds from multi-view camera imagery. RadarGen adapts efficient image-latent diffusion to the radar domain by representing radar measurements in bird's-eye-view form that encodes spatial structure together with radar cross section (RCS) and Doppler attributes. A lightweight recovery step reconstructs point clouds from the generated maps. To better align generation with the visual scene, RadarGen incorporates BEV-aligned depth, semantic, and motion cues extracted from pretrained foundation models, which guide the stochastic generation process toward physically plausible radar patterns. Conditioning on images makes the approach broadly compatible, in principle, with existing visual datasets and simulation frameworks, offering a scalable direction for multimodal generative simulation. Evaluations on large-scale driving data show that RadarGen captures characteristic radar measurement distributions and reduces the gap to perception models trained on real data, marking a step toward unified generative simulation across sensing modalities.

---

## 4. Keypoint Counting Classifiers: Turning Vision Transformers into Self-Explainable Models Without Training

**论文链接:** [http://arxiv.org/abs/2512.17891v1](http://arxiv.org/abs/2512.17891v1)

**作者:** Kristoffer Wickstrøm, Teresa Dorszewski, Siyan Chen, Michael Kampffmeyer, Elisabeth Wetzer, Robert Jenssen

**发布时间:** 2025-12-19

### GPT解析

### 总结

论文提出了一种名为关键点计数分类器(KCCs)的新方法，可将任何训练好的基于视觉Transformer(ViT)的模型转换为自解释模型(SEM)，无需重新训练，从而提高模型的透明度和可靠性。

### 背景

当前设计自解释模型的方法需要复杂的训练流程和特定架构，不切实际。随着基于视觉Transformer的通用基础模型的发展，这种不切实际性更加严重，需要新方法提供透明度和可靠性。

### 目的

开发一种无需重新训练即可将任何基于ViT的模型转换为自解释模型的方法，以提高ViT基础模型的透明度和可靠性。

### 方法

提出关键点计数分类器(KCCs)方法，利用ViT能够高精度自动识别图像间匹配关键点的特性，创建一个易于解释且可在输入中可视化的决策过程。

### 主要发现

广泛评估表明，KCCs比最近的基线方法改善了人机通信效果。

### 结论

KCCs是使基于ViT的基础模型更加透明和可靠的重要一步。

### 翻译

当前设计自解释模型(SEMs)的方法需要复杂的训练流程和特定的架构，这使得它们不切实际。随着基于视觉Transformer(ViT)的通用基础模型的进步，这种不切实际性变得更加严重。因此，需要新的方法为基于ViT的基础模型提供透明度和可靠性。在这项工作中，我们提出了一种新方法，可以将任何训练好的基于ViT的模型转换为SEM，而无需重新训练，我们称之为关键点计数分类器(KCCs)。最近的研究表明，ViT可以高精度地自动识别图像之间的匹配关键点，我们基于这些结果创建了一个易于解释的决策过程，该过程在输入中本质上是可可视化的。我们进行了广泛的评估，表明与最近的基线相比，KCCs改善了人机通信。我们相信KCCs是使基于ViT的基础模型更加透明和可靠的重要一步。


### 论文摘要

Current approaches for designing self-explainable models (SEMs) require complicated training procedures and specific architectures which makes them impractical. With the advance of general purpose foundation models based on Vision Transformers (ViTs), this impracticability becomes even more problematic. Therefore, new methods are necessary to provide transparency and reliability to ViT-based foundation models. In this work, we present a new method for turning any well-trained ViT-based model into a SEM without retraining, which we call Keypoint Counting Classifiers (KCCs). Recent works have shown that ViTs can automatically identify matching keypoints between images with high precision, and we build on these results to create an easily interpretable decision process that is inherently visualizable in the input. We perform an extensive evaluation which show that KCCs improve the human-machine communication compared to recent baselines. We believe that KCCs constitute an important step towards making ViT-based foundation models more transparent and reliable.

---

## 5. AnyTask: an Automated Task and Data Generation Framework for Advancing Sim-to-Real Policy Learning

**论文链接:** [http://arxiv.org/abs/2512.17853v1](http://arxiv.org/abs/2512.17853v1)

**作者:** Ran Gong, Xiaohan Zhang, Jinghuan Shang, Maria Vittoria Minniti, Jigarkumar Patel, Valerio Pepe, Riedana Yan, Ahmet Gundogdu, Ivan Kapelyukh, Ali Abbas, Xiaoqiang Yan, Harsh Patel, Laura Herlant, Karl Schmeckpeper

**发布时间:** 2025-12-19

**备注:** 28 pages, 25 figures. The first four authors contributed equally

### GPT解析

### 总结

这篇论文介绍了AnyTask框架，这是一个结合大规模并行GPU模拟和基础模型的自动化框架，用于设计多样化的操作任务并合成机器人数据。研究人员提出了三种AnyTask代理来生成专家演示，并在生成的数据上训练行为克隆策略，最终在真实机器人硬件上部署，成功实现了44%的平均成功率。

### 背景

通用机器人学习仍然受到数据的限制：大规模、多样化且高质量的交互数据在现实世界中收集成本高昂。虽然模拟已成为扩大数据收集规模的有前途的方法，但相关任务（包括模拟任务设计、任务感知场景生成、专家演示合成和模拟到现实迁移）仍然需要大量的人力投入。

### 目的

开发一个自动化框架，减少机器人学习过程中对大规模、多样化且高质量交互数据的依赖，降低数据收集成本，并减少人工干预。

### 方法

提出AnyTask框架，结合大规模并行GPU模拟和基础模型。引入三种AnyTask代理：1) ViPR，一种新颖的任务和运动规划代理，具有循环并行细化；2) ViPR-Eureka，一种具有生成密集奖励和LLM引导接触采样的强化学习代理；3) ViPR-RL，一种混合规划和学习方法，仅使用稀疏奖励共同产生高质量演示。

### 主要发现

训练的行为克隆策略在模拟中得到验证，并直接部署在真实机器人硬件上。这些策略能够泛化到新的物体姿态，在一套真实世界的抓放、抽屉开启、接触丰富的推动和长期操作任务中实现了44%的平均成功率。

### 结论

AnyTask框架能够自动化设计多样化的操作任务并合成机器人数据，减少了对大规模、高质量人工收集数据的依赖，并在真实世界任务中取得了良好的性能。

### 翻译

通用机器人学习仍然受到数据的限制：大规模、多样化且高质量的交互数据在现实世界中收集成本高昂。虽然模拟已成为扩大数据收集规模的有前途的方法，但相关任务，包括模拟任务设计、任务感知场景生成、专家演示合成和模拟到现实迁移，仍然需要大量的人力投入。我们提出了AnyTask，一个将大规模并行GPU模拟与基础模型配对的自动化框架，用于设计多样化的操作任务并合成机器人数据。我们引入了三种AnyTask代理，用于生成专家演示，旨在解决尽可能多的任务：1) ViPR，一种新颖的任务和运动规划代理，具有循环并行细化；2) ViPR-Eureka，一种具有生成密集奖励和LLM引导接触采样的强化学习代理；3) ViPR-RL，一种混合规划和学习方法，仅使用稀疏奖励共同产生高质量演示。我们在生成的数据上训练行为克隆策略，在模拟中验证它们，并将它们直接部署在真实机器人硬件上。这些策略能够泛化到新的物体姿态，在一套真实世界的抓放、抽屉开启、接触丰富的推动和长期操作任务中实现了44%的平均成功率。我们的项目网站是https://anytask.rai-inst.com。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决机器人学习中的数据瓶颈问题：大规模、多样化、高质量的机器人交互数据在现实世界中收集极其昂贵和耗时。这个问题很重要，因为机器人学习成功依赖于大量高质量数据，而现实世界数据收集成本高、效率低；虽然模拟可以扩展数据收集，但任务设计和数据生成仍需大量人工干预，限制了数据的多样性和机器人系统的能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到基础模型（如大型语言模型）在机器人任务中表现出色，可以利用它们来自动化创建机器人模拟环境的关键步骤。设计上结合了大规模并行GPU模拟和基础模型，从高层次文本目标自动生成任务和数据。该方法借鉴了现有工作：使用任务和运动规划（TAMP）和强化学习（RL）两种常用方法，利用IsaacLab模拟器，并参考了基础模型的能力，但试图解决现有系统在自动化程度和sim-to-real转移方面的局限性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用大规模并行GPU模拟和基础模型（特别是大型语言模型和视觉语言模型）来自动化整个机器人数据生成流程，从高层次文本目标开始，自动生成任务、场景和专家演示数据。整体流程包括：1)构建对象数据库存储物体信息；2)任务生成器提出任务和对象；3)模拟生成器创建可执行代码；4)密集注释系统生成详细的环境描述；5)使用三种代理（VIPR、VIPR-EUREKA和VIPR-RL）生成专家演示；6)在生成的数据上训练策略并直接部署到真实机器人。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)完全自动化的端到端框架；2)三种多样化代理生成专家演示；3)密集注释系统；4)基于网格的接触采样算法；5)两阶段数据收集机制。相比之前工作，不同之处在于：实现了更高程度的自动化；生成任务描述具有更好多样性；实现了纯合成数据训练的零样本sim-to-real转移；利用大规模GPU并行处理提高效率；结合了TAMP和RL的优点，而非单独使用其中一种。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'AnyTask框架通过自动化任务设计、场景生成和专家演示创建，利用大规模GPU模拟和基础模型，实现了从纯合成数据到零样本sim-to-real策略学习的完整流程，显著减少了机器人学习中对真实世界数据的依赖。'}


### 论文摘要

Generalist robot learning remains constrained by data: large-scale, diverse, and high-quality interaction data are expensive to collect in the real world. While simulation has become a promising way for scaling up data collection, the related tasks, including simulation task design, task-aware scene generation, expert demonstration synthesis, and sim-to-real transfer, still demand substantial human effort. We present AnyTask, an automated framework that pairs massively parallel GPU simulation with foundation models to design diverse manipulation tasks and synthesize robot data. We introduce three AnyTask agents for generating expert demonstrations aiming to solve as many tasks as possible: 1) ViPR, a novel task and motion planning agent with VLM-in-the-loop Parallel Refinement; 2) ViPR-Eureka, a reinforcement learning agent with generated dense rewards and LLM-guided contact sampling; 3) ViPR-RL, a hybrid planning and learning approach that jointly produces high-quality demonstrations with only sparse rewards. We train behavior cloning policies on generated data, validate them in simulation, and deploy them directly on real robot hardware. The policies generalize to novel object poses, achieving 44% average success across a suite of real-world pick-and-place, drawer opening, contact-rich pushing, and long-horizon manipulation tasks. Our project website is at https://anytask.rai-inst.com .

---

## 6. Chorus: Multi-Teacher Pretraining for Holistic 3D Gaussian Scene Encoding

**论文链接:** [http://arxiv.org/abs/2512.17817v1](http://arxiv.org/abs/2512.17817v1)

**作者:** Yue Li, Qi Ma, Runyi Yang, Mengjiao Ma, Bin Ren, Nikola Popovic, Nicu Sebe, Theo Gevers, Luc Van Gool, Danda Pani Paudel, Martin R. Oswald

**发布时间:** 2025-12-19

### GPT解析

### 总结

Chorus是一个多教师预训练框架，通过从2D基础模型中提炼互补信号，学习一个整体的3D高斯溅射场景编码器，解决了直接从3DGS基元编码丰富通用特征的研究空白。

### 背景

3DGS已成为高保真场景表示方法，但直接从其基元编码丰富、通用特征的研究仍不充分。

### 目的

开发一个框架，能够从3DGS基元中学习丰富的、通用的特征表示。

### 方法

Chorus使用共享的3D编码器和教师特定的投影仪，从语言对齐、通用和物体感知的教师那里学习，鼓励捕获从高级语义到细粒度结构的共享嵌入空间。

### 主要发现

1) Chorus在多种任务上表现出色，包括开放词汇的语义和实例分割等；2) 在点云基准上，使用少39.9倍训练场景的情况下优于点云基线；3) 提出的渲染和提炼适应方法有助于领域外微调。

### 结论

Chorus成功解决了从3DGS基元编码丰富通用特征的研究空白，并在多种任务和基准上展示了优越性能。

### 翻译

虽然3DGS已成为一种高保真场景表示方法，但直接从其基元编码丰富、通用特征的研究仍不充分。我们通过引入Chorus（一个多教师预训练框架）解决了这一研究空白，该框架通过从2D基础模型中提炼互补信号，学习一个整体的、前馈的3D高斯溅射场景编码器。Chorus使用共享的3D编码器和教师特定的投影仪，从语言对齐、通用和物体感知的教师那里学习，鼓励捕获从高级语义到细粒度结构的共享嵌入空间。我们在广泛任务上评估了Chorus：开放词汇的语义和实例分割、线性和解码器探测，以及数据高效监督。除了3DGS外，我们还通过仅使用高斯的中心、颜色和估计法线作为输入预训练了一个变体，在仅支持点云的几个基准上测试了Chorus。有趣的是，这个编码器展示了强大的迁移能力，并且使用少39.9倍训练场景的情况下优于点云基线。最后，我们提出了一种渲染和提炼的适应方法，有助于领域外微调。我们的代码和模型将在发表后发布。


### 论文摘要

While 3DGS has emerged as a high-fidelity scene representation, encoding rich, general-purpose features directly from its primitives remains under-explored. We address this gap by introducing Chorus, a multi-teacher pretraining framework that learns a holistic feed-forward 3D Gaussian Splatting (3DGS) scene encoder by distilling complementary signals from 2D foundation models. Chorus employs a shared 3D encoder and teacher-specific projectors to learn from language-aligned, generalist, and object-aware teachers, encouraging a shared embedding space that captures signals from high-level semantics to fine-grained structure.   We evaluate Chorus on a wide range of tasks: open-vocabulary semantic and instance segmentation, linear and decoder probing, as well as data-efficient supervision. Besides 3DGS, we also test Chorus on several benchmarks that only support point clouds by pretraining a variant using only Gaussians' centers, colors, estimated normals as inputs. Interestingly, this encoder shows strong transfer and outperforms the point clouds baseline while using 39.9 times fewer training scenes. Finally, we propose a render-and-distill adaptation that facilitates out-of-domain finetuning. Our code and model will be released upon publication.

---

## 7. Intelligent Knowledge Mining Framework: Bridging AI Analysis and Trustworthy Preservation

**论文链接:** [http://arxiv.org/abs/2512.17795v1](http://arxiv.org/abs/2512.17795v1)

**作者:** Binh Vu

**发布时间:** 2025-12-19

### GPT解析

### 总结

本文介绍了智能知识挖掘框架(IKMF)，一个全面的概念模型，旨在弥合动态AI驱动分析与可信长期保存之间的关键差距。

### 背景

数字数据的空前增长给所有数据密集型部门带来了访问、整合和价值创造的挑战。有价值的信息经常被封装在分散的系统、非结构化文档和异构格式中，形成了阻碍有效利用和协作决策的信息孤岛。

### 目的

设计一个概念模型，连接动态AI驱动分析与可信长期保存，解决数据孤岛问题，促进数据的有效利用和协作决策。

### 方法

提出双流架构：水平挖掘流程将原始数据转换为语义丰富、机器可操作的知识；并行的可信归档流确保这些资产的完整性、来源和计算可重现性。

### 主要发现

通过定义这种共生关系的蓝图，为将静态存储库转变为促进可操作智能从生产者流向消费者的活跃生态系统提供了基础模型。

### 结论

本文概述了指导框架研发的动机、问题陈述和关键研究问题，介绍了潜在的科学方法论，并详细说明了其概念设计和建模。

### 翻译

数字数据的空前增长给所有数据密集型部门在访问、整合和价值创造方面带来了重大挑战。有价值的信息经常被封装在分散的系统、非结构化文档和异构格式中，形成了阻碍有效利用和协作决策的信息孤岛。本文介绍了智能知识挖掘框架(IKMF)，一个全面的概念模型，旨在弥合动态AI驱动分析与可信长期保存之间的关键差距。该框架提出了双流架构：一个水平挖掘流程，系统地将原始数据转换为语义丰富、机器可操作的知识；以及一个并行的可信归档流，确保这些资产的完整性、来源和计算可重现性。通过定义这种共生关系的蓝图，本文为将静态存储库转变为促进可操作智能从生产者流向消费者的活跃生态系统提供了基础模型。本文概述了指导框架研发的动机、问题陈述和关键研究问题，介绍了潜在的科学方法论，并详细说明了其概念设计和建模。


### 论文摘要

The unprecedented proliferation of digital data presents significant challenges in access, integration, and value creation across all data-intensive sectors. Valuable information is frequently encapsulated within disparate systems, unstructured documents, and heterogeneous formats, creating silos that impede efficient utilization and collaborative decision-making. This paper introduces the Intelligent Knowledge Mining Framework (IKMF), a comprehensive conceptual model designed to bridge the critical gap between dynamic AI-driven analysis and trustworthy long-term preservation. The framework proposes a dual-stream architecture: a horizontal Mining Process that systematically transforms raw data into semantically rich, machine-actionable knowledge, and a parallel Trustworthy Archiving Stream that ensures the integrity, provenance, and computational reproducibility of these assets. By defining a blueprint for this symbiotic relationship, the paper provides a foundational model for transforming static repositories into living ecosystems that facilitate the flow of actionable intelligence from producers to consumers. This paper outlines the motivation, problem statement, and key research questions guiding the research and development of the framework, presents the underlying scientific methodology, and details its conceptual design and modeling.

---

## 8. ClothHMR: 3D Mesh Recovery of Humans in Diverse Clothing from Single Image

**论文链接:** [http://arxiv.org/abs/2512.17545v1](http://arxiv.org/abs/2512.17545v1)

**作者:** Yunqi Gao, Leyuan Liu, Yuhan Li, Changxin Gao, Yuanyuan Liu, Jingying Chen

**发布时间:** 2025-12-19

**DOI:** 10.1145/3731715.3733288

**备注:** 15 pages,16 figures

### GPT解析

### 总结

本文提出了ClothHMR方法，通过服装剪裁和基于基础人体视觉模型的网格恢复两个模块，准确恢复穿着各种服装的人体3D网格，解决了现有方法在处理宽松服装时表现不佳的问题。

### 背景

随着3D数据作为多媒体信息的重要形式迅速涌现，3D人体网格恢复技术也相应发展。然而，当前方法主要处理穿着紧身服装的人体，在估计各种服装（特别是宽松服装）下的人体形状和姿态时表现不佳。

### 目的

为了准确恢复穿着各种服装的人体3D网格，作者提出了两个关键见解：(1)将服装剪裁以适合人体可以减轻服装对3D人体网格恢复的不利影响；(2)利用来自大型基础模型的人体视觉信息可以增强估计的泛化能力。

### 方法

ClothHMR主要包含两个模块：1. 服装剪裁(CT)模块：采用身体语义估计和身体边缘预测来剪裁服装，确保其适合身体轮廓；2. 基于FHVM的网格恢复(MR)模块：通过不断将3D网格的中间表示与从基础人体视觉模型(FHVM)推断出的中间表示对齐，来优化3D人体网格的初始参数。

### 主要发现

实验结果表明，ClothHMR在基准数据集和野外图像上显著优于现有的最先进方法。此外，开发了一个由ClothHMR驱动的在线时尚购物网络应用，展示了ClothHMR可以有效服务实际使用场景。

### 结论

ClothHMR能够准确恢复穿着各种服装的人体3D网格，精确估计他们的身体形状和姿态。该方法的代码和模型已在GitHub上公开。

### 翻译

随着3D数据迅速成为多媒体信息的重要形式，3D人体网格恢复技术也相应发展。然而，当前方法主要处理穿着紧身服装的人体，在估计各种服装（特别是宽松服装）下的人体形状和姿态时表现不佳。为此，我们提出了两个关键见解：(1)将服装剪裁以适合人体可以减轻服装对3D人体网格恢复的不利影响，(2)利用来自大型基础模型的人体视觉信息可以增强估计的泛化能力。基于这些见解，我们提出了ClothHMR，以准确恢复穿着各种服装的人体3D网格。ClothHMR主要包含两个模块：服装剪裁(CT)和基于FHVM的网格恢复(MR)。CT模块采用身体语义估计和身体边缘预测来剪裁服装，确保其适合身体轮廓。MR模块通过不断将3D网格的中间表示与从基础人体视觉模型(FHVM)推断出的中间表示对齐，来优化3D人体网格的初始参数。ClothHMR能够准确恢复穿着各种服装的人体3D网格，精确估计他们的身体形状和姿态。实验结果表明，ClothHMR在基准数据集和野外图像上显著优于现有的最先进方法。此外，开发了一个由ClothHMR驱动的在线时尚购物网络应用，展示了ClothHMR可以有效服务实际使用场景。ClothHMR的代码和模型可在以下网址获取：https://github.com/starVisionTeam/ClothHMR。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D人体网格恢复技术在处理穿着多样化衣物（尤其是宽松衣物）时效果不佳的问题。这个问题在现实中很重要，因为准确的3D人体网格恢复在虚拟试衣、在线健身、沉浸式游戏等领域有广泛应用，而人们日常穿着各种类型的衣物，特别是宽松衣物非常常见。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于两个关键洞察：1) 将衣物裁剪以适合人体轮廓可以减轻衣物对3D人体网格恢复的不利影响；2) 利用大型基础模型中的人体视觉信息可以增强估计的泛化能力。作者设计了ClothHMR方法，包括服装裁剪和基于FHVM的网格恢复两个模块。作者借鉴了SMPL等参数化人体模型、Sapiens基础视觉模型、迭代优化策略以及特征金字塔融合等技术。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过服装裁剪减少衣物遮挡的影响，并使用单一基础模型提取一致的中间表示，然后通过迭代优化对齐不同表示。整体流程是：输入单张RGB图像→服装裁剪模块（身体语义估计、边缘预测和衣物裁剪）→网格恢复模块（初始化3D网格→基础模型提取中间表示→迭代优化网格参数）→输出精确的3D人体网格。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出服装裁剪方案处理多样化衣物；2) 首次将基础人体视觉模型引入3D人体网格恢复；3) 提出ClothHMR方法恢复穿着多样化衣物的3D网格。相比之前工作，ClothHMR通过服装裁剪减少遮挡影响，使用单一基础模型确保中间表示一致性，并在多个基准数据集上显著优于现有方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ClothHMR通过服装裁剪和基础人体视觉模型的结合，实现了从单张图像中准确恢复穿着多样化衣物的人体3D网格，显著提高了在宽松衣物和复杂姿势下的估计精度。'}


### 论文摘要

With 3D data rapidly emerging as an important form of multimedia information, 3D human mesh recovery technology has also advanced accordingly. However, current methods mainly focus on handling humans wearing tight clothing and perform poorly when estimating body shapes and poses under diverse clothing, especially loose garments. To this end, we make two key insights: (1) tailoring clothing to fit the human body can mitigate the adverse impact of clothing on 3D human mesh recovery, and (2) utilizing human visual information from large foundational models can enhance the generalization ability of the estimation. Based on these insights, we propose ClothHMR, to accurately recover 3D meshes of humans in diverse clothing. ClothHMR primarily consists of two modules: clothing tailoring (CT) and FHVM-based mesh recovering (MR). The CT module employs body semantic estimation and body edge prediction to tailor the clothing, ensuring it fits the body silhouette. The MR module optimizes the initial parameters of the 3D human mesh by continuously aligning the intermediate representations of the 3D mesh with those inferred from the foundational human visual model (FHVM). ClothHMR can accurately recover 3D meshes of humans wearing diverse clothing, precisely estimating their body shapes and poses. Experimental results demonstrate that ClothHMR significantly outperforms existing state-of-the-art methods across benchmark datasets and in-the-wild images. Additionally, a web application for online fashion and shopping powered by ClothHMR is developed, illustrating that ClothHMR can effectively serve real-world usage scenarios. The code and model for ClothHMR are available at: \url{https://github.com/starVisionTeam/ClothHMR}.

---

## 9. SafeBench-Seq: A Homology-Clustered, CPU-Only Baseline for Protein Hazard Screening with Physicochemical/Composition Features and Cluster-Aware Confidence Intervals

**论文链接:** [http://arxiv.org/abs/2512.17527v1](http://arxiv.org/abs/2512.17527v1)

**作者:** Muhammad Haris Khan

**发布时间:** 2025-12-19

### GPT解析

### 总结

SafeBench-Seq是一个仅包含元数据、可重现的蛋白质序列危险筛查基准测试和基线分类器，在同源控制下评估，可在普通CPU上运行。

### 背景

蛋白质设计基础模型带来了具体的生物安全风险，但社区缺乏一种简单、可重现的序列级别危险筛查基线，这种基线需要在同源控制下明确评估，并且可以在普通CPU上运行。

### 目的

引入SafeBench-Seq，一个仅包含元数据、可重现的基准测试和基线分类器，完全基于公共数据（SafeProtein危险数据和UniProt良性数据）和可解释特征（全局物理化学描述符和氨基酸组成）构建。

### 方法

通过同源聚类（<=40%同一性）组合数据集模拟'前所未见'的威胁；执行聚类级别的保留（训练/测试间无聚类重叠）；报告区分度（AUROC/AUPRC）和筛查操作点（TPR@1% FPR；FPR@95% TPR）及95%自举置信区间；提供校准概率；使用Brier分数、期望校准误差和可靠性图表量化概率质量；通过保持组成的残基洗牌和仅长度/组成的消融探测快捷键易感性。

### 主要发现

随机分割相对于同源聚类评估显著高估了稳健性；校准的线性模型表现出相对较好的校准性，而树集成保持略高的Brier/ECE。

### 结论

SafeBench-Seq仅使用CPU、可重现，并且仅发布元数据（访问号、聚类ID、分割标签），能够在不传播危险序列的情况下进行严格评估。

### 翻译

蛋白质设计基础模型带来了具体的生物安全风险，但社区缺乏一种简单、可重现的序列级别危险筛查基线，这种基线需要在同源控制下明确评估，并且可以在普通CPU上运行。我们引入了SafeBench-Seq，这是一个仅包含元数据、可重现的基准测试和基线分类器，完全基于公共数据（SafeProtein危险数据和UniProt良性数据）和可解释特征（全局物理化学描述符和氨基酸组成）构建。为了模拟'前所未见'的威胁，我们在<=40%同一性下对组合数据集进行同源聚类，并执行聚类级别的保留（训练/测试之间无聚类重叠）。我们报告了区分度（AUROC/AUPRC）和筛查操作点（TPR@1% FPR；FPR@95% TPR），并附有95%的自举置信区间（n=200），我们通过CalibratedClassifierCV（逻辑回归/随机森林使用等距；线性SVM使用Platt Sigmoid）提供校准概率。我们使用Brier分数、期望校准误差（ECE；15个区间）和可靠性图表来量化概率质量。通过保持组成的残基洗牌和仅长度/组成的消融来探测快捷键易感性。经验上，与同源聚类评估相比，随机分割显著高估了稳健性；校准的线性模型表现出相对较好的校准性，而树集成保持略高的Brier/ECE。SafeBench-Seq仅使用CPU、可重现，并且仅发布元数据（访问号、聚类ID、分割标签），能够在不传播危险序列的情况下进行严格评估。


### 论文摘要

Foundation models for protein design raise concrete biosecurity risks, yet the community lacks a simple, reproducible baseline for sequence-level hazard screening that is explicitly evaluated under homology control and runs on commodity CPUs. We introduce SafeBench-Seq, a metadata-only, reproducible benchmark and baseline classifier built entirely from public data (SafeProtein hazards and UniProt benigns) and interpretable features (global physicochemical descriptors and amino-acid composition). To approximate "never-before-seen" threats, we homology-cluster the combined dataset at <=40% identity and perform cluster-level holdouts (no cluster overlap between train/test). We report discrimination (AUROC/AUPRC) and screening-operating points (TPR@1% FPR; FPR@95% TPR) with 95% bootstrap confidence intervals (n=200), and we provide calibrated probabilities via CalibratedClassifierCV (isotonic for Logistic Regression / Random Forest; Platt sigmoid for Linear SVM). We quantify probability quality using Brier score, Expected Calibration Error (ECE; 15 bins), and reliability diagrams. Shortcut susceptibility is probed via composition-preserving residue shuffles and length-/composition-only ablations. Empirically, random splits substantially overestimate robustness relative to homology-clustered evaluation; calibrated linear models exhibit comparatively good calibration, while tree ensembles retain slightly higher Brier/ECE. SafeBench-Seq is CPU-only, reproducible, and releases metadata only (accessions, cluster IDs, split labels), enabling rigorous evaluation without distributing hazardous sequences.

---

## 10. Foundation Model Priors Enhance Object Focus in Feature Space for Source-Free Object Detection

**论文链接:** [http://arxiv.org/abs/2512.17514v1](http://arxiv.org/abs/2512.17514v1)

**作者:** Sairam VCR, Rishabh Lalla, Aveen Dayal, Tejal Kulkarni, Anuj Lalla, Vineeth N Balasubramanian, Muhammad Haris Khan

**发布时间:** 2025-12-19

### GPT解析

### 总结

论文提出了一种名为FALCON-SFOD的框架，用于增强域偏移下的目标聚焦适应，解决了现有方法在无源目标检测中因域偏移导致的目标表示能力下降问题。

### 背景

当前无源目标检测的最先进方法通常依赖Mean-Teacher自标记，但域偏移会降低检测器保持强目标聚焦表示的能力，导致在背景杂乱中产生高置信度激活，进而生成不可靠的伪标签。

### 目的

提出一种框架，加强特征空间本身，增强目标聚焦表示能力，以解决域偏移下目标检测性能下降的问题。

### 方法

提出FALCON-SFOD框架，包含两个互补组件：1) SPAR（空间先验感知正则化）：利用视觉基础模型的泛化能力正则化检测器特征空间，通过OV-SAM派生的类无关二值掩码引导网络朝向目标区域；2) IRPL（不平衡感知噪声鲁棒伪标记）：促进严重前景-背景不平衡下的平衡和噪声容忍学习。

### 主要发现

域偏移导致检测器目标聚焦能力下降，现有方法主要关注伪标签的改进而忽略了特征空间本身的加强；通过理论分析将组件设计与更紧密的定位和分类误差边界联系起来。

### 结论

FALCON-SFOD框架通过加强目标聚焦表示和特征空间，在SFOD基准测试中取得了具有竞争力的性能。

### 翻译

当前无源目标检测的最先进方法通常依赖于Mean-Teacher自标记。然而，域偏移往往会降低检测器保持强目标聚焦表示的能力，导致在背景杂乱中产生高置信度激活。这种弱目标聚焦导致检测头产生不可靠的伪标签。虽然之前的工作主要改进这些伪标签，但它们忽略了加强特征空间本身的基本需求。我们提出了FALCON-SFOD（具有杂波抑制和噪声鲁棒性的基础对齐学习），一个旨在加强域偏移下目标聚焦适应的框架。它包含两个互补组件。SPAR（空间先验感知正则化）利用视觉基础模型的泛化能力来正则化检测器的特征空间。使用从OV-SAM派生的类无关二值掩码，SPAR通过引导网络朝向目标区域来促进结构化和前景聚焦的激活。IRPL（不平衡感知噪声鲁棒伪标记）通过在严重前景-背景不平衡下促进平衡和噪声容忍学习来补充SPAR。在将这些设计与更紧密的定位和分类误差边界联系起来的理论分析指导下，FALCON-SFOD在SFOD基准测试中取得了具有竞争力的性能。


### 论文摘要

Current state-of-the-art approaches in Source-Free Object Detection (SFOD) typically rely on Mean-Teacher self-labeling. However, domain shift often reduces the detector's ability to maintain strong object-focused representations, causing high-confidence activations over background clutter. This weak object focus results in unreliable pseudo-labels from the detection head. While prior works mainly refine these pseudo-labels, they overlook the underlying need to strengthen the feature space itself. We propose FALCON-SFOD (Foundation-Aligned Learning with Clutter suppression and Noise robustness), a framework designed to enhance object-focused adaptation under domain shift. It consists of two complementary components. SPAR (Spatial Prior-Aware Regularization) leverages the generalization strength of vision foundation models to regularize the detector's feature space. Using class-agnostic binary masks derived from OV-SAM, SPAR promotes structured and foreground-focused activations by guiding the network toward object regions. IRPL (Imbalance-aware Noise Robust Pseudo-Labeling) complements SPAR by promoting balanced and noise-tolerant learning under severe foreground-background imbalance. Guided by a theoretical analysis that connects these designs to tighter localization and classification error bounds, FALCON-SFOD achieves competitive performance across SFOD benchmarks.

---

## 11. Validation of Diagnostic Artificial Intelligence Models for Prostate Pathology in a Middle Eastern Cohort

**论文链接:** [http://arxiv.org/abs/2512.17499v1](http://arxiv.org/abs/2512.17499v1)

**作者:** Peshawa J. Muhammad Ali, Navin Vincent, Saman S. Abdulla, Han N. Mohammed Fadhl, Anders Blilie, Kelvin Szolnoky, Julia Anna Mielcarz, Xiaoyi Ji, Nita Mulliqi, Abdulbasit K. Al-Talabani, Kimmo Kartasalo

**发布时间:** 2025-12-19

**备注:** 40 pages, 8 figures, 11 tables

### GPT解析

### 总结

该研究展示了第一个来自中东地区的外部验证队列，评估了AI在前列腺癌诊断和Gleason分级中的性能，证明了AI模型达到病理学家水平的表现，并展示了低成本扫描仪在AI病理学应用中的可行性。

### 背景

人工智能正在提高癌症诊断的效率和准确性，但病理学AI系统的性能几乎只在欧美大型中心的人群中进行了评估。为了在全球范围内采用病理学AI，需要对代表性不足的人群进行验证研究，这些人群从AI支持中获得的潜在收益可能最大。

### 目的

展示第一个来自中东地区的外部验证队列研究，专注于基于AI的前列腺癌诊断和Gleason分级。

### 方法

收集并数字化了来自伊拉克库尔德地区的339个前列腺活检标本，代表2013-2024年间连续的185例患者。评估了一个任务特定的端到端AI模型和两个基础模型，分析它们与病理学家的一致性和在不同扫描仪型号上数字化样本的一致性。

### 主要发现

AI与病理学家之间的分级一致性类似于病理学家之间的分级一致性（Cohen二次加权kappa值0.801 vs 0.799，p=0.9824）。所有AI模型和扫描仪对的跨扫描仪一致性都很高（二次加权kappa > 0.90），包括低成本紧凑型扫描仪。

### 结论

AI模型在前列腺组织病理学评估中表现出与病理学家相当的性能。紧凑型扫描仪可为非数字化环境中的验证研究提供途径，并使样本量有限的实验室能够经济有效地采用AI。这是中东第一个公开可用的数字病理数据集，支持进一步研究全球公平的AI病理学。

### 翻译

背景：人工智能正在提高癌症诊断的效率和准确性。病理学AI系统的性能几乎只在来自大型中心的欧美队列上进行了评估。为了在全球范围内采用病理学AI，需要对目前代表性不足的人群进行验证研究，这些人群从AI支持中获得的潜在收益可能最大。我们展示了第一个来自中东的外部验证队列研究，专注于基于AI的前列腺癌诊断和Gleason分级。方法：我们收集并数字化了来自伊拉克库尔德地区的339个前列腺活检标本，代表2013-2024年间连续的185例患者。我们评估了一个任务特定的端到端AI模型和两个基础模型，在它们与病理学家的一致性和在三种扫描仪型号（Hamamatsu、Leica和Grundium）上数字化样本的一致性方面。发现：AI与病理学家之间的分级一致性类似于病理学家之间的分级一致性，Cohen二次加权kappa值为0.801 vs 0.799（p=0.9824）。所有AI模型和扫描仪对的跨扫描仪一致性都很高（二次加权kappa > 0.90），包括低成本紧凑型扫描仪。结论：AI模型在前列腺组织病理学评估中表现出与病理学家相当的性能。紧凑型扫描仪可为非数字化环境中的验证研究提供途径，并使样本量有限的实验室能够经济有效地采用AI。这是中东第一个公开可用的数字病理数据集，支持进一步研究全球公平的AI病理学。资金：SciLifeLab和Wallenberg数据驱动生命科学项目，Instrumentarium科学基金会，卡罗林斯卡研究所研究基金会。


### 论文摘要

Background: Artificial intelligence (AI) is improving the efficiency and accuracy of cancer diagnostics. The performance of pathology AI systems has been almost exclusively evaluated on European and US cohorts from large centers. For global AI adoption in pathology, validation studies on currently under-represented populations - where the potential gains from AI support may also be greatest - are needed. We present the first study with an external validation cohort from the Middle East, focusing on AI-based diagnosis and Gleason grading of prostate cancer.   Methods: We collected and digitised 339 prostate biopsy specimens from the Kurdistan region, Iraq, representing a consecutive series of 185 patients spanning the period 2013-2024. We evaluated a task-specific end-to-end AI model and two foundation models in terms of their concordance with pathologists and consistency across samples digitised on three scanner models (Hamamatsu, Leica, and Grundium).   Findings: Grading concordance between AI and pathologists was similar to pathologist-pathologist concordance with Cohen's quadratically weighted kappa 0.801 vs. 0.799 (p=0.9824). Cross-scanner concordance was high (quadratically weighted kappa > 0.90) for all AI models and scanner pairs, including low-cost compact scanner.   Interpretation: AI models demonstrated pathologist-level performance in prostate histopathology assessment. Compact scanners can provide a route for validation studies in non-digitalised settings and enable cost-effective adoption of AI in laboratories with limited sample volumes. This first openly available digital pathology dataset from the Middle East supports further research into globally equitable AI pathology.   Funding: SciLifeLab and Wallenberg Data Driven Life Science Program, Instrumentarium Science Foundation, Karolinska Institutet Research Foundation.

---

## 12. MMLANDMARKS: a Cross-View Instance-Level Benchmark for Geo-Spatial Understanding

**论文链接:** [http://arxiv.org/abs/2512.17492v1](http://arxiv.org/abs/2512.17492v1)

**作者:** Oskar Kristoffersen, Alba R. Sánchez, Morten R. Hannemose, Anders B. Dahl, Dim P. Papadopoulos

**发布时间:** 2025-12-19

### GPT解析

### 总结

本文介绍了一个名为MMLANDMARKS的多模态地标数据集，包含四种模态：高分辨率航空图像、地面视角图像、文本信息和地理坐标，用于支持多种地理空间分析任务。

### 背景

地理空间分析受益于多模态方法，因为每个地理位置可以通过多种方式描述。然而，当前地理空间基准在模态覆盖方面有限，限制了该领域的进展，因为现有方法无法在统一框架内整合所有相关模态。

### 目的

引入MMLANDMARKS数据集，这是一个包含四种模态的基准数据集，用于训练和评估各种地理空间任务的模型。

### 方法

MMLANDMARKS数据集包含197k高分辨率航空图像、329k地面视角图像、文本信息和美国18,557个不同地标的地理坐标，每个模态间有一一对应关系，支持跨视图地面到卫星检索、地面和卫星地理定位、文本到图像和文本到GPS检索等任务。

### 主要发现

通过采用一个简单的受CLIP启发的基线方法，作者展示了在不同任务上的广泛泛化能力和与现成基础模型和专门的最先进模型相比的竞争性能。

### 结论

实现广泛的地理空间理解需要多模态数据集的支持。

### 翻译

地理空间分析通过多模态方法受益，因为每个地理位置都可以通过多种方式描述（不同视角的图像、文本描述和地理坐标）。当前的地理空间基准在模态覆盖方面有限，显著限制了该领域的进展，因为当前方法无法在统一框架内整合所有相关模态。我们引入了多模态地标数据集（MMLANDMARKS），这是一个由四种模态组成的基准：197k高分辨率航空图像、329k地面视角图像、文本信息以及美国18,557个不同地标的地理坐标。MMLANDMARKS数据集在每个模态上都具有一一对应关系，能够为各种地理空间任务（包括跨视图地面到卫星检索、地面和卫星地理定位、文本到图像和文本到GPS检索）的训练和基准测试模型。通过采用一个简单的受CLIP启发的基线方法，我们展示了在不同任务上的广泛泛化能力和与现成基础模型和专门的最先进模型相比的竞争性能，这说明了实现广泛地理空间理解需要多模态数据集的必要性。


### 论文摘要

Geo-spatial analysis of our world benefits from a multimodal approach, as every single geographic location can be described in numerous ways (images from various viewpoints, textual descriptions, and geographic coordinates). Current geo-spatial benchmarks have limited coverage across modalities, considerably restricting progress in the field, as current approaches cannot integrate all relevant modalities within a unified framework. We introduce the Multi-Modal Landmark dataset (MMLANDMARKS), a benchmark composed of four modalities: 197k highresolution aerial images, 329k ground-view images, textual information, and geographic coordinates for 18,557 distinct landmarks in the United States. The MMLANDMARKS dataset has a one-to-one correspondence across every modality, which enables training and benchmarking models for various geo-spatial tasks, including cross-view Ground-to-Satellite retrieval, ground and satellite geolocalization, Text-to-Image, and Text-to-GPS retrieval. We demonstrate broad generalization and competitive performance against off-the-shelf foundational models and specialized state-of-the-art models across different tasks by employing a simple CLIP-inspired baseline, illustrating the necessity for multimodal datasets to achieve broad geo-spatial understanding.

---

## 13. 论文ID: 2512.17491v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.17491v1.json'

---

## 14. Any-Optical-Model: A Universal Foundation Model for Optical Remote Sensing

**论文链接:** [http://arxiv.org/abs/2512.17224v1](http://arxiv.org/abs/2512.17224v1)

**作者:** Xuyang Li, Chenyu Li, Danfeng Hong

**发布时间:** 2025-12-19

**备注:** Accepted by AAAI2026

### GPT解析

### 总结

本研究提出了Any Optical Model (AOM)，一种通用的遥感基础模型，能够处理不同光学卫星的任意波段组成、传感器类型和分辨率尺度，解决了现有模型在缺失波段、跨传感器融合和跨分辨率场景下的局限性。

### 背景

光学卫星具有多样化的波段布局和地面采样距离，为生态系统监测到应急响应等多种任务提供关键证据。然而，不同光学传感器的波段组成和空间分辨率差异显著，现有的遥感基础模型通常在固定配置下预训练，难以应对现实世界中的复杂场景。

### 目的

开发一种能够适应任意波段组成、传感器类型和分辨率尺度的通用遥感基础模型，提高模型在现实应用中的泛化能力和实用性。

### 方法

AOM采用三种关键技术：1) 光谱无关的tokenizer为每个通道分配专门波段嵌入，明确编码光谱特性；2) 多尺度自适应块嵌入机制动态调整感受野，捕获从亚米到百米尺度的纹理和上下文模式；3) 多尺度语义对齐机制结合通道级自监督掩码重建预训练策略，共同建模光谱-空间关系并保持全局语义一致性。

### 主要发现

在超过10个公共数据集（包括Sentinel-2、Landsat和HLS数据）上的广泛实验表明，AOM在缺失波段、跨传感器和跨分辨率设置等挑战性条件下持续达到最先进(SOTA)性能。

### 结论

AOM成功解决了现有遥感基础模型在处理不同光学卫星数据时的局限性，为遥感基础模型在现实世界应用中的泛化能力和实际部署提供了有效解决方案。

### 翻译

光学卫星凭借其多样的波段布局和地面采样距离，为从生态系统监测到应急响应等任务提供了不可或缺的证据。然而，不同光学传感器在波段组成和空间分辨率上的显著差异，给现有的遥感基础模型(RSFMs)带来了主要挑战。这些模型通常在固定的波段配置和分辨率上预训练，在面对涉及缺失波段、跨传感器融合和未见空间尺度的现实场景时容易受到影响，从而限制了它们的泛化能力和实际部署。为解决这些局限性，我们提出了Any Optical Model (AOM)，一种专门设计用于适应任意波段组成、传感器类型和分辨率尺度的通用RSFM。为在缺失或新增波段时保持独特的光谱特性，AOM引入了一种与光谱无关的tokenizer，为每个通道分配专门的波段嵌入，实现光谱身份的明确编码。为有效捕获从亚米到百米图像的纹理和上下文模式，我们设计了一种多尺度自适应块嵌入机制，可动态调整感受野。此外，为在不同分辨率间保持全局语义一致性，AOM结合了多尺度语义对齐机制和通道级自监督掩码重建预训练策略，共同建模光谱-空间关系。在Sentinel-2、Landsat和HLS等超过10个公共数据集上的广泛实验表明，AOM在缺失波段、跨传感器和跨分辨率设置等挑战条件下持续取得了最先进(SOTA)的性能。


### 论文摘要

Optical satellites, with their diverse band layouts and ground sampling distances, supply indispensable evidence for tasks ranging from ecosystem surveillance to emergency response. However, significant discrepancies in band composition and spatial resolution across different optical sensors present major challenges for existing Remote Sensing Foundation Models (RSFMs). These models are typically pretrained on fixed band configurations and resolutions, making them vulnerable to real world scenarios involving missing bands, cross sensor fusion, and unseen spatial scales, thereby limiting their generalization and practical deployment. To address these limitations, we propose Any Optical Model (AOM), a universal RSFM explicitly designed to accommodate arbitrary band compositions, sensor types, and resolution scales. To preserve distinctive spectral characteristics even when bands are missing or newly introduced, AOM introduces a spectrum-independent tokenizer that assigns each channel a dedicated band embedding, enabling explicit encoding of spectral identity. To effectively capture texture and contextual patterns from sub-meter to hundred-meter imagery, we design a multi-scale adaptive patch embedding mechanism that dynamically modulates the receptive field. Furthermore, to maintain global semantic consistency across varying resolutions, AOM incorporates a multi-scale semantic alignment mechanism alongside a channel-wise self-supervised masking and reconstruction pretraining strategy that jointly models spectral-spatial relationships. Extensive experiments on over 10 public datasets, including those from Sentinel-2, Landsat, and HLS, demonstrate that AOM consistently achieves state-of-the-art (SOTA) performance under challenging conditions such as band missing, cross sensor, and cross resolution settings.

---

## 15. Biosecurity-Aware AI: Agentic Risk Auditing of Soft Prompt Attacks on ESM-Based Variant Predictors

**论文链接:** [http://arxiv.org/abs/2512.17146v1](http://arxiv.org/abs/2512.17146v1)

**作者:** Huixin Zhan

**发布时间:** 2025-12-19

### GPT解析

### 总结

研究团队引入了安全代理基因组评估器(SAGE)，用于审计基因组基础模型(GFMs)的对抗性漏洞，发现即使是最先进的模型如ESM2也容易受到软提示攻击的影响。

### 背景

基因组基础模型(GFMs)如进化尺度建模(ESM)在变异效应预测方面表现出色，但它们在对抗性操纵下的安全性和鲁棒性尚未得到充分探索。

### 目的

引入安全代理基因组评估器(SAGE)，一个用于审计GFMs对抗性漏洞的代理框架，以解决基因组基础模型安全性研究空白的问题。

### 方法

SAGE通过可解释和自动化的风险审计循环运行，注入软提示扰动，监控模型在训练检查点上的行为，计算AUROC和AUPR等风险指标，并使用基于大型语言模型的叙述解释生成结构化报告，从而在不修改底层模型的情况下持续评估嵌入空间的鲁棒性。

### 主要发现

即使是最先进的GFMs如ESM2也容易受到针对软提示攻击的影响，这些攻击会导致可测量的性能下降，揭示了基因组基础模型中关键且先前隐藏的漏洞。

### 结论

代理风险审计对于确保临床变异解释等生物医学应用的安全性至关重要。

### 翻译

基因组基础模型（GFMs），如进化尺度建模（ESM），在变异效应预测方面已经显示出显著的成功。然而，它们在对抗性操纵下的安全性和鲁棒性在很大程度上仍未被探索。为了解决这一空白，我们引入了安全代理基因组评估器（SAGE），这是一个用于审计GFMs对抗性漏洞的代理框架。SAGE通过可解释和自动化的风险审计循环运行。它注入软提示扰动，监控模型在训练检查点上的行为，计算AUROC和AUPR等风险指标，并使用基于大型语言模型的叙述解释生成结构化报告。这种代理过程能够在不修改底层模型的情况下持续评估嵌入空间的鲁棒性。使用SAGE，我们发现即使是像ESM2这样的最先进的GFMs也容易受到针对软提示攻击的影响，导致可测量的性能下降。这些发现揭示了基因组基础模型中关键且先前隐藏的漏洞，表明在保障临床变异解释等生物医学应用中，代理风险审计的重要性。


### 论文摘要

Genomic Foundation Models (GFMs), such as Evolutionary Scale Modeling (ESM), have demonstrated remarkable success in variant effect prediction. However, their security and robustness under adversarial manipulation remain largely unexplored. To address this gap, we introduce the Secure Agentic Genomic Evaluator (SAGE), an agentic framework for auditing the adversarial vulnerabilities of GFMs. SAGE functions through an interpretable and automated risk auditing loop. It injects soft prompt perturbations, monitors model behavior across training checkpoints, computes risk metrics such as AUROC and AUPR, and generates structured reports with large language model-based narrative explanations. This agentic process enables continuous evaluation of embedding-space robustness without modifying the underlying model. Using SAGE, we find that even state-of-the-art GFMs like ESM2 are sensitive to targeted soft prompt attacks, resulting in measurable performance degradation. These findings reveal critical and previously hidden vulnerabilities in genomic foundation models, showing the importance of agentic risk auditing in securing biomedical applications such as clinical variant interpretation.

---

## 16. SDUM: A Scalable Deep Unrolled Model for Universal MRI Reconstruction

**论文链接:** [http://arxiv.org/abs/2512.17137v1](http://arxiv.org/abs/2512.17137v1)

**作者:** Puyang Wang, Pengfei Guo, Keyi Chai, Jinyuan Zhou, Daguang Xu, Shanshan Jiang

**发布时间:** 2025-12-19

### GPT解析

### 总结

论文提出了可扩展深度展开模型(SDUM)，一个通用的MRI深度学习重建框架，能够在多种成像协议上实现高性能重建，无需针对特定任务进行微调。

### 背景

临床MRI包含多种成像协议，涵盖不同的解剖目标、对比度、采样模式和加速因子，而当前深度学习重建方法通常是特定于协议的，限制了其泛化和部署能力。

### 目的

开发一个通用的MRI深度学习重建框架，能够处理多种不同的成像协议，实现高性能重建。

### 方法

提出可扩展深度展开模型(SDUM)，这是一个通用框架，包含基于Restormer的重建器、学习的线圈敏感度图估计器(CSME)、采样感知的加权数据一致性(SWDC)、级联索引和协议元数据的通用条件化(UC)以及渐进式级联扩展训练。

### 主要发现

SDUM表现出类似基础模型的扩展行为；单个SDUM在异构数据上训练后，在CMRxRecon2025挑战赛的四个赛道上达到最先进结果，比专用基线高出最多1.0 dB；在CMRxRecon2024上比获胜方法高出0.55 dB；在fastMRI大脑上超过PC-RNN，高出1.8 dB；消融研究验证了各组件的有效性。

### 结论

SDUM为通用、可扩展的MRI重建提供了一条实用路径。

### 翻译

临床MRI包含多种成像协议，涵盖不同的解剖目标（心脏、大脑、膝盖）、对比度（T1、T2、mapping）、采样模式（笛卡尔、径向、螺旋、kt空间）和加速因子，然而当前深度学习重建通常是特定于协议的，阻碍了泛化和部署。我们引入了可扩展深度展开模型（SDUM），这是一个通用框架，结合了基于Restormer的重建器、学习的线圈敏感度图估计器（CSME）、采样感知的加权数据一致性（SWDC）、级联索引和协议元数据的通用条件化（UC）以及渐进式级联扩展训练。SDUM表现出类似基础模型的扩展行为：重建质量随参数的对数增长而提高，相关系数r=0.986（R²=0.973），最多到18个级联，证明了随着模型深度的性能增益可预测。单个在异构数据上训练的SDUM无需任务特定的微调，在CMRxRecon2025挑战赛的所有四个赛道上实现了最先进的结果——多中心、多疾病、5T和儿科，比专用基线高出最多1.0 dB。在CMRxRecon2024上，SDUM比获胜方法PromptMR+高出0.55 dB；在fastMRI大脑上，它超过了PC-RNN，高出1.8 dB。消融研究验证了每个组件的贡献：SWDC比标准DC高出0.43 dB，每个级联的CSME高出0.51 dB，UC高出0.38 dB。这些结果确立了SDUM作为通用、可扩展MRI重建的实用路径。


### 论文摘要

Clinical MRI encompasses diverse imaging protocols--spanning anatomical targets (cardiac, brain, knee), contrasts (T1, T2, mapping), sampling patterns (Cartesian, radial, spiral, kt-space), and acceleration factors--yet current deep learning reconstructions are typically protocol-specific, hindering generalization and deployment. We introduce Scalable Deep Unrolled Model (SDUM), a universal framework combining a Restormer-based reconstructor, a learned coil sensitivity map estimator (CSME), sampling-aware weighted data consistency (SWDC), universal conditioning (UC) on cascade index and protocol metadata, and progressive cascade expansion training. SDUM exhibits foundation-model-like scaling behavior: reconstruction quality follows PSNR ${\sim}$ log(parameters) with correlation $r{=}0.986$ ($R^2{=}0.973$) up to 18 cascades, demonstrating predictable performance gains with model depth. A single SDUM trained on heterogeneous data achieves state-of-the-art results across all four CMRxRecon2025 challenge tracks--multi-center, multi-disease, 5T, and pediatric--without task-specific fine-tuning, surpassing specialized baselines by up to ${+}1.0$~dB. On CMRxRecon2024, SDUM outperforms the winning method PromptMR+ by ${+}0.55$~dB; on fastMRI brain, it exceeds PC-RNN by ${+}1.8$~dB. Ablations validate each component: SWDC ${+}0.43$~dB over standard DC, per-cascade CSME ${+}0.51$~dB, UC ${+}0.38$~dB. These results establish SDUM as a practical path toward universal, scalable MRI reconstruction.

---

## 17. Sigma-MoE-Tiny Technical Report

**论文链接:** [http://arxiv.org/abs/2512.16248v2](http://arxiv.org/abs/2512.16248v2)

**作者:** Qingguo Hu, Zhenghao Lin, Ziyue Yang, Yucheng Ding, Xiao Liu, Yuting Jiang, Ruizhe Wang, Tianyu Chen, Zhongxin Guo, Yifan Xiong, Rui Gao, Lei Qu, Jinsong Su, Peng Cheng, Yeyun Gong

**发布时间:** 2025-12-18

### GPT解析

### 总结

本文介绍了Sigma-MoE-Tiny，一种实现最高稀疏性的Mixture-of-Experts语言模型，通过细粒度专家分割和渐进式稀疏化调度解决了专家负载平衡问题，尽管只激活0.5B参数，仍能实现与更大规模模型相当的性能。

### 背景

Mixture-of-Experts (MoE)已成为基础模型的一种有前途的范式，因其高效和强大的可扩展性。然而，MoE模型面临的主要挑战是专家负载平衡问题，特别是在极端稀疏的情况下。

### 目的

开发一种实现最高稀疏性的MoE语言模型，解决极端稀疏性带来的专家负载平衡问题，并在保持训练稳定性的同时平衡专家利用率。

### 方法

采用细粒度专家分割，每层最多96个专家，为每个token只激活一个专家，实现20B总参数中仅0.5B激活；提出渐进式稀疏化调度来解决负载平衡问题；在多样化和高质量语料上进行预训练，然后进行后训练以进一步释放能力。

### 主要发现

广泛使用的负载平衡损失在低层设置下变得无效；渐进式稀疏化调度能有效平衡专家利用率和训练稳定性；整个训练过程保持稳定，没有出现不可恢复的损失尖峰；尽管只激活0.5B参数，Sigma-MoE-Tiny仍能在可比或显著更大规模的同类模型中实现顶级性能。

### 结论

Sigma-MoE-Tiny证明了在极端稀疏条件下实现高性能MoE模型的可能性；对高度稀疏MoE模型中负载平衡的深入讨论为未来MoE架构的稀疏性进步提供了见解。

### 翻译

混合专家模型已成为基础模型的一种有前途的范式，因其高效和强大的可扩展性。在本工作中，我们提出了Sigma-MoE-Tiny，一种MoE语言模型，与现有开源模型相比实现了最高的稀疏性。Sigma-MoE-Tiny采用细粒度专家分割，每层最多96个专家，同时只为每个token激活一个专家，从而在20B总参数中仅激活0.5B。这种极端稀疏性带来的主要挑战在于专家负载平衡。我们发现，在这种设置下，广泛使用的负载平衡损失在较低层 tends to become ineffective。为解决这一问题，我们提出了一种渐进式稀疏化调度，旨在平衡专家利用率和训练稳定性。Sigma-MoE-Tiny在多样化和高质量语料上进行预训练，随后进行后训练以进一步释放其能力。整个训练过程保持显著稳定，没有发生不可恢复的损失尖峰。全面评估显示，尽管仅激活0.5B参数，Sigma-MoE-Tiny在可比或显著更大规模的同类模型中实现了顶级性能。此外，我们对高度稀疏MoE模型中的负载平衡进行了深入讨论，为未来MoE架构的稀疏性进步提供了见解。项目页面：https://qghuxmu.github.io/Sigma-MoE-Tiny 代码：https://github.com/microsoft/ltp-megatron-lm


### 论文摘要

Mixture-of-Experts (MoE) has emerged as a promising paradigm for foundation models due to its efficient and powerful scalability. In this work, we present Sigma-MoE-Tiny, an MoE language model that achieves the highest sparsity compared to existing open-source models. Sigma-MoE-Tiny employs fine-grained expert segmentation with up to 96 experts per layer, while activating only one expert for each token, resulting in 20B total parameters with just 0.5B activated. The major challenge introduced by such extreme sparsity lies in expert load balancing. We find that the widely-used load balancing loss tends to become ineffective in the lower layers under this setting. To address this issue, we propose a progressive sparsification schedule aiming to balance expert utilization and training stability. Sigma-MoE-Tiny is pre-trained on a diverse and high-quality corpus, followed by post-training to further unlock its capabilities. The entire training process remains remarkably stable, with no occurrence of irrecoverable loss spikes. Comprehensive evaluations reveal that, despite activating only 0.5B parameters, Sigma-MoE-Tiny achieves top-tier performance among counterparts of comparable or significantly larger scale. In addition, we provide an in-depth discussion of load balancing in highly sparse MoE models, offering insights for advancing sparsity in future MoE architectures.   Project page: https://qghuxmu.github.io/Sigma-MoE-Tiny   Code: https://github.com/microsoft/ltp-megatron-lm

---

## 18. 论文ID: 2512.17012v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.17012v1.json'

---

## 19. MedNeXt-v2: Scaling 3D ConvNeXts for Large-Scale Supervised Representation Learning in Medical Image Segmentation

**论文链接:** [http://arxiv.org/abs/2512.17774v1](http://arxiv.org/abs/2512.17774v1)

**作者:** Saikat Roy, Yannick Kirchhoff, Constantin Ulrich, Maximillian Rokuss, Tassilo Wald, Fabian Isensee, Klaus Maier-Hein

**发布时间:** 2025-12-19

### GPT解析

### 总结

本研究提出了一种新的3D医学图像分割骨干网络MedNeXt-v2，通过改进微架构和数据缩放策略，实现了在大规模监督预训练下的最先进性能。研究还发现，更强的骨干网络在相似数据上表现更好，表示缩放对病理分割的益处更大，且模态特定预训练在完全微调后几乎没有额外好处。

### 背景

大规模监督预训练正在迅速改变3D医学图像分割领域，但现有工作主要专注于增加数据集规模，忽略了骨干网络在大规模时是否是有效的表示学习器这一问题。

### 目的

重新审视用于体积分割的ConvNeXt架构，引入MedNeXt-v2这一复合缩放的3D ConvNeXt，利用改进的微架构和数据缩放来实现最先进的性能。

### 方法

首先证明常用骨干网络在大规模预训练中通常不是最优的；然后在扩展前进行全面的骨干网络基准测试，证明更强的从头开始性能可靠地预测预训练后的下游性能；结合3D全局响应归一化模块，使用深度、宽度和上下文缩放改进架构；在18k个体积CT上预训练MedNeXt-v2，并在六个CT和MR基准测试上进行微调。

### 主要发现

MedNeXt-v2在六个具有挑战性的CT和MR基准测试(144个结构)上展示了最先进的性能，比七个公开发布的预训练模型有显著改进；更强的骨干网络在相似数据上产生更好结果；表示缩放不成比例地有利于病理分割；模态特定预训练在完全微调后几乎没有额外好处。

### 结论

MedNeXt-v2是3D医学图像分割中大规模监督表示学习的强大骨干网络，相关代码和预训练模型已公开在官方nnUNet存储库中。

### 翻译

大规模监督预训练正在迅速改变3D医学图像分割。然而，现有工作主要专注于增加数据集规模，而忽略了骨干网络在大规模时是否是有效的表示学习器这一问题。在这项工作中，我们通过重新审视用于体积分割的ConvNeXt架构来解决这个问题，并引入了MedNeXt-v2，这是一种复合缩放的3D ConvNeXt，利用改进的微架构和数据缩放来实现最先进的性能。首先，我们证明大规模预训练流程中常用的骨干网络通常不是最优的。随后，我们在扩展前使用全面的骨干网络基准测试，并证明更强的从头开始性能可靠地预测预训练后的更强下游性能。在这些发现的指导下，我们结合了3D全局响应归一化模块，并使用深度、宽度和上下文缩放来改进我们的架构，以实现有效的表示学习。我们在18k个体积CT上预训练MedNeXt-v2，并在六个具有挑战性的CT和MR基准测试(144个结构)上进行微调时展示了最先进的性能，比七个公开发布的预训练模型显示出一致的改进。除了改进之外，我们对这些模型的基准测试还揭示更强的骨干网络在相似数据上产生更好的结果，表示缩放不成比例地有利于病理分割，并且一旦应用完全微调，模态特定的预训练几乎没有好处。总之，我们的研究结果表明MedNeXt-v2是3D医学图像分割中大规模监督表示学习的强大骨干网络。我们的代码和预训练模型可在官方nnUNet存储库中获取：https://www.github.com/MIC-DKFZ/nnUNet

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决的问题是：在3D医学图像分割的大规模监督预训练中，研究者往往只关注增加数据集规模，却忽视了骨干网络是否适合作为有效的表示学习器。这个问题很重要，因为不合适的骨干网络会限制大规模数据的利用效率，影响下游分割任务的性能，导致计算资源浪费和模型效果不佳。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有大规模预训练方法的三个系统性问题：使用遗留的骨干架构、未在验证骨干网络前扩展数据、仅与非预训练基线比较。通过系统性实验，他们发现更强的骨干网络在预训练后能带来更好的下游性能。基于这一发现，作者借鉴了ConvNeXt架构和EfficientNet的复合缩放理念，在MedNeXt基础上添加了3D全局响应归一化(GRN)模块，并采用了深度、宽度和上下文缩放策略，最终设计了MedNeXt-v2。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过改进骨干网络的微架构和合理缩放网络规模，结合大规模数据预训练，提高3D医学图像分割的性能。实现流程包括：1)在小规模数据集上验证多种骨干网络性能；2)在MedNeXt架构中添加3D GRN模块防止特征崩溃；3)应用深度、宽度和上下文缩放增加网络容量；4)在18k个体积的CT数据上进行预训练；5)在下游任务上微调，使用更大的输入上下文(192×192×192)提升性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)系统性验证骨干网络选择在大规模预训练中的重要性；2)引入3D全局响应归一化(GRN)模块提升表示学习效果；3)采用复合缩放策略同时优化深度、宽度和上下文；4)提出'小上下文预训练+大上下文微调'的策略平衡计算成本和性能；5)首次对七个公开发布的预训练模型进行系统性基准测试。相比之前工作，MedNeXt-v2更注重骨干网络质量而非仅关注数据规模，通过微架构改进和合理缩放实现了更高效的表示学习。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MedNeXt-v2通过改进3D ConvNeXt架构的微架构和复合缩放策略，结合大规模预训练，实现了3D医学图像分割的最先进性能，并首次系统性地评估了多种预训练模型的性能，为该领域提供了重要的见解和基准。'}


### 论文摘要

Large-scale supervised pretraining is rapidly reshaping 3D medical image segmentation. However, existing efforts focus primarily on increasing dataset size and overlook the question of whether the backbone network is an effective representation learner at scale. In this work, we address this gap by revisiting ConvNeXt-based architectures for volumetric segmentation and introducing MedNeXt-v2, a compound-scaled 3D ConvNeXt that leverages improved micro-architecture and data scaling to deliver state-of-the-art performance. First, we show that routinely used backbones in large-scale pretraining pipelines are often suboptimal. Subsequently, we use comprehensive backbone benchmarking prior to scaling and demonstrate that stronger from scratch performance reliably predicts stronger downstream performance after pretraining. Guided by these findings, we incorporate a 3D Global Response Normalization module and use depth, width, and context scaling to improve our architecture for effective representation learning. We pretrain MedNeXt-v2 on 18k CT volumes and demonstrate state-of-the-art performance when fine-tuning across six challenging CT and MR benchmarks (144 structures), showing consistent gains over seven publicly released pretrained models. Beyond improvements, our benchmarking of these models also reveals that stronger backbones yield better results on similar data, representation scaling disproportionately benefits pathological segmentation, and that modality-specific pretraining offers negligible benefit once full finetuning is applied. In conclusion, our results establish MedNeXt-v2 as a strong backbone for large-scale supervised representation learning in 3D Medical Image Segmentation. Our code and pretrained models are made available with the official nnUNet repository at: https://www.github.com/MIC-DKFZ/nnUNet

---

## 20. 论文ID: 2512.17577v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.17577v1.json'

---

## 21. Alzheimer's Disease Brain Network Mining

**论文链接:** [http://arxiv.org/abs/2512.17276v1](http://arxiv.org/abs/2512.17276v1)

**作者:** Alireza Moayedikia, Sara Fin

**发布时间:** 2025-12-19

### GPT解析

### 总结

MATCH-AD是一种半监督学习框架，通过整合深度表示学习、基于图的标签传播和最优传输理论，解决了阿尔茨海默病诊断中标签稀缺的问题，实现了接近完美的诊断准确率。

### 背景

阿尔茨海默病诊断的机器学习方法面临根本性挑战，临床评估昂贵且有创，导致神经影像数据集中只有小部分有真实标签。全球积累了大量部分注释的神经影像数据，但其诊断潜力尚未被充分利用。

### 目的

开发一种半监督学习框架，解决标签稀缺问题，利用神经影像数据中的流形结构，将有限标注样本的诊断信息传播到更大的未标注人群，并使用Wasserstein距离量化认知状态之间的疾病进展。

### 方法

提出了Multi view Adaptive Transport Clustering for Heterogeneous Alzheimer's Disease (MATCH-AD)框架，整合了深度表示学习、基于图的标签传播和最优传输理论。在近5000名国家阿尔茨海默病协调中心受试者上进行了评估，使用了来自数百个脑区域的结构MRI测量、脑脊液生物标志物和临床变量。

### 主要发现

尽管真实标签不足三分之一，MATCH-AD实现了接近完美的诊断准确率；框架显著优于所有基线方法，实现了几乎完美的一致性(kappa)，而最佳基线方法仅有微弱一致性；即使在严重标签稀缺的情况下，性能仍保持临床有用性；提供了理论收敛保证，并证明了标签传播误差和传输稳定性的界限。

### 结论

原则性的半监督学习可以解锁全球积累的大量部分注释神经影像数据的诊断潜力，大大减少了注释负担，同时保持了适合临床部署的准确性。

### 翻译

阿尔茨海默病(AD)诊断的机器学习方法面临根本性挑战。临床评估昂贵且有创，导致神经影像数据集中只有小部分有真实标签。我们引入了针对异质性阿尔茨海默病的多视图自适应传输聚类(MATCH-AD)，这是一个半监督框架，整合了深度表示学习、基于图的标签传播和最优传输理论来解决这个问题。该框架利用神经影像数据中的流形结构，将有限标注样本的诊断信息传播到更大的未标注人群，同时使用Wasserstein距离量化认知状态之间的疾病进展。在国家阿尔茨海默病协调中心近5000名受试者上进行了评估，包括来自数百个脑区域的结构MRI测量、脑脊液生物标志物和临床变量，尽管真实标签不足三分之一，MATCH-AD仍实现了接近完美的诊断准确率。该框架显著优于所有基线方法，与最佳基线方法的微弱一致性相比，实现了表示几乎完美一致性的kappa值，这是诊断可靠性的质的转变。即使在严重的标签稀缺情况下，性能仍保持临床有用性，我们提供了理论收敛保证，并证明了标签传播误差和传输稳定性的界限。这些结果表明，原则性的半监督学习可以解锁全球积累的大量部分注释神经影像数据的诊断潜力，大大减少注释负担，同时保持适合临床部署的准确性。


### 论文摘要

Machine learning approaches for Alzheimer's disease (AD) diagnosis face a fundamental challenges. Clinical assessments are expensive and invasive, leaving ground truth labels available for only a fraction of neuroimaging datasets. We introduce Multi view Adaptive Transport Clustering for Heterogeneous Alzheimer's Disease (MATCH-AD), a semi supervised framework that integrates deep representation learning, graph-based label propagation, and optimal transport theory to address this limitation. The framework leverages manifold structure in neuroimaging data to propagate diagnostic information from limited labeled samples to larger unlabeled populations, while using Wasserstein distances to quantify disease progression between cognitive states. Evaluated on nearly five thousand subjects from the National Alzheimer's Coordinating Center, encompassing structural MRI measurements from hundreds of brain regions, cerebrospinal fluid biomarkers, and clinical variables MATCHAD achieves near-perfect diagnostic accuracy despite ground truth labels for less than one-third of subjects. The framework substantially outperforms all baseline methods, achieving kappa indicating almost perfect agreement compared to weak agreement for the best baseline, a qualitative transformation in diagnostic reliability. Performance remains clinically useful even under severe label scarcity, and we provide theoretical convergence guarantees with proven bounds on label propagation error and transport stability. These results demonstrate that principled semi-supervised learning can unlock the diagnostic potential of the vast repositories of partially annotated neuroimaging data accumulating worldwide, substantially reducing annotation burden while maintaining accuracy suitable for clinical deployment.

---

## 22. A Theoretical Analysis of State Similarity Between Markov Decision Processes

**论文链接:** [http://arxiv.org/abs/2512.17265v1](http://arxiv.org/abs/2512.17265v1)

**作者:** Zhenyu Tao, Wei Xu, Xiaohu You

**发布时间:** 2025-12-19

**备注:** Submitted to an IEEE Transactions. arXiv admin note: substantial text overlap with arXiv:2509.18714

### GPT解析

### 总结

这项工作提出了广义双模拟度量(GBSM)，解决了多个马尔可夫决策过程(MDP)之间状态相似性测量的挑战。GBSM具有严格的数学基础，并在策略转移、状态聚合和基于采样的估计等多个应用中表现出色。

### 背景

双模拟度量(BSM)是分析马尔可夫决策过程中状态相似性的有力工具，已被成功用于强化学习中的状态表征学习和策略探索。然而，BSM应用于多个MDP之间的状态相似性仍然具有挑战性，先前的工作虽试图扩展BSM到成对MDP，但缺乏完善的数学性质限制了进一步的理论分析。

### 目的

正式建立一种广义双模拟度量(GBSM)，用于测量任意成对MDP之间的状态相似性，并为其提供严格的数学基础。

### 方法

严格证明GBSM具有三个基本度量性质：对称性、跨MDP三角不等式和相同空间上的距离界限。利用这些性质理论上分析跨MDP的策略转移、状态聚合和基于采样的估计，提供比基于标准BSM的现有界限更严格的显式界限，并给出改进的样本复杂度估计。

### 主要发现

GBSM被严格证明具有三个基本度量性质；理论分析获得了比现有方法更严格的界限；提供了改进的样本复杂度估计；数值结果验证了理论发现并证明了GBSM在多MDP场景中的有效性。

### 结论

广义双模拟度量(GBSM)为多个MDP之间的状态相似性测量提供了严格的数学基础和有效的工具，在策略转移、状态聚合和基于采样的估计等多个应用中表现出色。

### 翻译

双模拟度量(BSM)是分析马尔可夫决策过程(MDP)内状态相似性的有力工具，揭示了在BSM中更接近的状态具有更相似的最优值函数。虽然BSM已在强化学习(RL)中被成功用于状态表征学习和策略探索等任务，但其应用于多个MDP之间的状态相似性仍然具有挑战性。先前的工作试图将BSM扩展到成对的MDP，但缺乏完善的数学性质限制了MDP之间进一步的理论分析。在这项工作中，我们正式建立了一种广义双模拟度量(GBSM)，用于测量任意成对MDP之间的状态相似性，并通过三个基本度量性质严格证明了GBSM，即GBSM对称性、跨MDP三角不等式和相同空间上的距离界限。利用这些性质，我们理论上分析了跨MDP的策略转移、状态聚合和基于采样的估计，获得了比从标准BSM推导出的现有界限更严格的显式界限。此外，GBSM为估计提供了闭式样本复杂度，改进了基于BSM的现有渐近结果。数值结果验证了我们的理论发现，并证明了GBSM在多MDP场景中的有效性。


### 论文摘要

The bisimulation metric (BSM) is a powerful tool for analyzing state similarities within a Markov decision process (MDP), revealing that states closer in BSM have more similar optimal value functions. While BSM has been successfully utilized in reinforcement learning (RL) for tasks like state representation learning and policy exploration, its application to state similarity between multiple MDPs remains challenging. Prior work has attempted to extend BSM to pairs of MDPs, but a lack of well-established mathematical properties has limited further theoretical analysis between MDPs. In this work, we formally establish a generalized bisimulation metric (GBSM) for measuring state similarity between arbitrary pairs of MDPs, which is rigorously proven with three fundamental metric properties, i.e., GBSM symmetry, inter-MDP triangle inequality, and a distance bound on identical spaces. Leveraging these properties, we theoretically analyze policy transfer, state aggregation, and sampling-based estimation across MDPs, obtaining explicit bounds that are strictly tighter than existing ones derived from the standard BSM. Additionally, GBSM provides a closed-form sample complexity for estimation, improving upon existing asymptotic results based on BSM. Numerical results validate our theoretical findings and demonstrate the effectiveness of GBSM in multi-MDP scenarios.

---

## 23. SHARP-QoS: Sparsely-gated Hierarchical Adaptive Routing for joint Prediction of QoS

**论文链接:** [http://arxiv.org/abs/2512.17262v1](http://arxiv.org/abs/2512.17262v1)

**作者:** Suraj Kumar, Arvind Kumar, Soumi Chattopadhyay

**发布时间:** 2025-12-19

**备注:** 12 pages, 4 figures, 10 tables

### GPT解析

### 总结

本文提出了一种名为SHARP-QoS的统一策略，用于联合预测多种服务质量(QoS)参数。该方法解决了现有方法在处理稀疏、嘈杂且具有层次依赖关系的QoS数据时面临的挑战，包括负迁移问题和表示学习不足的问题。

### 背景

可靠的服务导向计算依赖于多种服务质量(QoS)参数来评估服务最优性。然而，现实世界中的QoS数据极其稀疏、嘈杂，并且受到QoS交互、地理和网络级别因素产生的层次依赖关系的影响，这使得准确的QoS预测具有挑战性。现有方法通常分别预测每个QoS参数，需要多个相似模型，增加了计算成本并导致泛化能力差。尽管最近的联合QoS预测研究探索了共享架构，但由于不同QoS参数间不一致的数值范围导致的损失缩放问题而遭受负迁移，并且在表示学习方面也存在不足，导致精度下降。

### 目的

开发一种统一的联合QoS预测策略，解决现有方法中的负迁移问题和表示学习不足的问题，提高QoS预测的准确性和效率。

### 方法

SHARP-QoS采用三个组件解决这些问题：1) 双机制：通过在庞加莱球中定义的双曲卷积从QoS和上下文结构中提取层次特征；2) 自适应特征共享机制：允许在信息丰富的QoS和上下文信号之间进行特征交换，使用门控特征融合模块支持动态特征选择；3) 基于EMA的损失平衡策略：允许稳定的联合优化，从而减轻负迁移。

### 主要发现

在具有两个、三个和四个QoS参数的三个数据集上的评估表明，SHARP-QoS优于单任务和多任务基线。广泛的研究表明，该模型有效地解决了主要挑战，包括稀疏性、对异常值的鲁棒性和冷启动问题，同时保持适度的计算开销。

### 结论

SHARP-QoS能够进行可靠的联合QoS预测，解决了现有方法中的关键问题，并在多个数据集上展示了优越的性能。

### 翻译

可靠的服务导向计算依赖于多种服务质量(QoS)参数，这些参数对于评估服务最优性至关重要。然而，现实世界中的QoS数据极其稀疏、嘈杂，并且受到QoS交互、地理和网络级别因素产生的层次依赖关系的影响，这使得准确的QoS预测具有挑战性。现有方法通常分别预测每个QoS参数，需要多个相似模型，这增加了计算成本并导致泛化能力差。尽管最近的联合QoS预测研究探索了共享架构，但由于不同QoS参数间不一致的数值范围导致的损失缩放问题而遭受负迁移，并且在表示学习方面也存在不足，导致精度下降。本文提出了一种名为SHARP-QoS的联合QoS预测统一策略，它使用三个组件解决了这些问题。首先，我们引入了一种双机制，通过在庞加莱球中定义的双曲卷积从QoS和上下文结构中提取层次特征。其次，我们提出了一种自适应特征共享机制，允许在信息丰富的QoS和上下文信号之间进行特征交换。采用门控特征融合模块支持在结构化和共享表示之间的动态特征选择。第三，我们设计了一种基于EMA的损失平衡策略，允许稳定的联合优化，从而减轻负迁移。在具有两个、三个和四个QoS参数的三个数据集上的评估表明，SHARP-QoS优于单任务和多任务基线。广泛的研究表明，我们的模型有效地解决了主要挑战，包括稀疏性、对异常值的鲁棒性和冷启动问题，同时保持适度的计算开销，凸显了其进行可靠的联合QoS预测的能力。


### 论文摘要

Dependable service-oriented computing relies on multiple Quality of Service (QoS) parameters that are essential to assess service optimality. However, real-world QoS data are extremely sparse, noisy, and shaped by hierarchical dependencies arising from QoS interactions, and geographical and network-level factors, making accurate QoS prediction challenging. Existing methods often predict each QoS parameter separately, requiring multiple similar models, which increases computational cost and leads to poor generalization. Although recent joint QoS prediction studies have explored shared architectures, they suffer from negative transfer due to loss-scaling caused by inconsistent numerical ranges across QoS parameters and further struggle with inadequate representation learning, resulting in degraded accuracy. This paper presents an unified strategy for joint QoS prediction, called SHARP-QoS, that addresses these issues using three components. First, we introduce a dual mechanism to extract the hierarchical features from both QoS and contextual structures via hyperbolic convolution formulated in the Poincaré ball. Second, we propose an adaptive feature-sharing mechanism that allows feature exchange across informative QoS and contextual signals. A gated feature fusion module is employed to support dynamic feature selection among structural and shared representations. Third, we design an EMA-based loss balancing strategy that allows stable joint optimization, thereby mitigating the negative transfer. Evaluations on three datasets with two, three, and four QoS parameters demonstrate that SHARP-QoS outperforms both single- and multi-task baselines. Extensive study shows that our model effectively addresses major challenges, including sparsity, robustness to outliers, and cold-start, while maintaining moderate computational overhead, underscoring its capability for reliable joint QoS prediction.

---

## 24. Disentangled representations via score-based variational autoencoders

**论文链接:** [http://arxiv.org/abs/2512.17127v1](http://arxiv.org/abs/2512.17127v1)

**作者:** Benjamin S. H. Lyo, Eero P. Simoncelli, Cristina Savin

**发布时间:** 2025-12-18

**备注:** 34 pages, 7 figures

### GPT解析

### 总结

本文提出了SAMI（基于评分的自编码器用于多尺度推断），一种结合扩散模型和VAE理论框架的无监督表征学习方法，通过统一证据下界制定目标函数，利用扩散过程评分指导学习表征，使隐式结构信息变得明确可解释。

### 背景

扩散模型和变分自编码器（VAE）是重要的无监督表征学习方法，但各自有不同的理论框架和应用特点，如何结合两者的优势是一个研究挑战。

### 目的

开发一种结合扩散模型和VAE理论框架的无监督表征学习方法，以学习能够自动捕捉数据中有意义结构的表征。

### 方法

提出SAMI方法，通过统一扩散模型和VAE的证据下界，制定基于原则的目标函数，利用底层扩散过程的评分指导来学习表征，并展示如何从预训练的扩散模型中提取有用表征。

### 主要发现

SAMI能恢复合成数据集的真实生成因子；从复杂自然图像中学习分解的、语义的潜在维度；将视频编码为比其他编码器更直的潜在轨迹，尽管仅在静态图像上训练；可从预训练扩散模型中提取有用表征；其显式概率性公式为识别语义轴提供了新方法。

### 结论

扩散模型中的隐式结构信息可以通过与VAE的协同组合变得明确且可解释，SAMI能够产生有意义的、结构化的表征，具有理论保证和实际应用价值。

### 翻译

我们提出了用于多尺度推断的基于评分的自编码器（SAMI），这是一种无监督表征学习方法，结合了扩散模型和VAE的理论框架。通过统一它们各自的证据下界，SAMI制定了一个基于原则的目标，通过底层扩散过程的评分指导来学习表征。得到的表征自动捕捉数据中的有意义结构：在我们的合成数据集中恢复真实生成因子，从复杂自然图像中学习分解的、语义的潜在维度，并将视频序列编码为比其他编码器更直的潜在轨迹，尽管仅在静态图像上训练。此外，SAMI可以从预训练的扩散模型中提取有用的表征，只需最少的额外训练。最后，显式概率性公式为在没有监督标签的情况下识别语义轴提供了新方法，其数学精确性使我们能够对学习表征的性质做出正式陈述。总体而言，这些结果表明，扩散模型中的隐式结构信息可以通过与变分自编码器的协同组合变得明确且可解释。


### 论文摘要

We present the Score-based Autoencoder for Multiscale Inference (SAMI), a method for unsupervised representation learning that combines the theoretical frameworks of diffusion models and VAEs. By unifying their respective evidence lower bounds, SAMI formulates a principled objective that learns representations through score-based guidance of the underlying diffusion process. The resulting representations automatically capture meaningful structure in the data: it recovers ground truth generative factors in our synthetic dataset, learns factorized, semantic latent dimensions from complex natural images, and encodes video sequences into latent trajectories that are straighter than those of alternative encoders, despite training exclusively on static images. Furthermore, SAMI can extract useful representations from pre-trained diffusion models with minimal additional training. Finally, the explicitly probabilistic formulation provides new ways to identify semantically meaningful axes in the absence of supervised labels, and its mathematical exactness allows us to make formal statements about the nature of the learned representation. Overall, these results indicate that implicit structural information in diffusion models can be made explicit and interpretable through synergistic combination with a variational autoencoder.

---

## 25. StereoMV2D: A Sparse Temporal Stereo-Enhanced Framework for Robust Multi-View 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2512.17620v1](http://arxiv.org/abs/2512.17620v1)

**作者:** Di Wu, Feng Yang, Wenhui Zhao, Jinwen Yu, Pan Liao, Benlian Xu, Dingwen Zhang

**发布时间:** 2025-12-19

**备注:** 12 pages, 4 figures. This work has been submitted to the IEEE for possible publication

### GPT解析

### 总结

StereoMV2D是一种统一框架，将时间立体建模集成到2D检测引导的多视角3D检测器中，通过利用相邻帧间同一物体的跨时间视差增强深度感知，优化查询先验，并在2D感兴趣区域内高效完成计算，实现了优越的检测性能。

### 背景

多视角3D目标检测是自动驾驶感知中的基础任务，平衡检测精度和计算效率至关重要。稀疏查询式3D检测器通过一组可学习查询高效聚合多视角图像中的目标相关特征，但单帧2D检测中的固有深度模糊限制了3D查询生成的准确性。

### 目的

解决单帧2D检测中的深度模糊问题，提高多视角3D目标检测的精度和召回率，同时保持计算效率。

### 方法

提出StereoMV2D框架，集成时间立体建模到2D检测引导的多视角3D检测器中；利用相邻帧间同一物体的跨时间视差增强深度感知；在2D感兴趣区域内高效完成计算；采用动态置信度门控机制评估时间立体线索的可靠性。

### 主要发现

通过利用跨时间视差，StereoMV2D能够增强深度感知并优化查询先验；动态置信度门控机制确保在物体外观变化和遮挡情况下的鲁棒检测；在nuScenes和Argoverse 2数据集上实现了优越的检测性能，且没有显著增加计算开销。

### 结论

StereoMV2D是一种有效的统一框架，能够提高多视角3D目标检测的准确性，同时保持计算效率，适用于自动驾驶感知任务。

### 翻译

多视角3D目标检测是自动驾驶感知中的基础任务，其中平衡检测精度和计算效率仍然至关重要。稀疏查询式3D检测器通过一组可学习查询高效地从多视角图像中聚合目标相关特征，提供了一种简洁且端到端的检测范式。基于这一基础，MV2D利用2D检测结果为查询初始化提供高质量的目标先验，实现更高的精度和召回率。然而，单帧2D检测中的固有深度模糊仍然限制了3D查询生成的准确性。为解决这一问题，我们提出了StereoMV2D，这是一个统一框架，将时间立体建模集成到2D检测引导的多视角3D检测器中。通过利用相邻帧间同一物体的跨时间视差，StereoMV2D增强了深度感知并优化了查询先验，同时在2D感兴趣区域(RoIs)内高效完成所有计算。此外，动态置信度门控机制通过学习来自帧间匹配矩阵的统计模式以及外观一致性，自适应地评估时间立体线索的可靠性，确保在物体外观和遮挡情况下的鲁棒检测。在nuScenes和Argoverse 2数据集上的大量实验表明，StereoMV2D实现了优越的检测性能，而没有带来显著的计算开销。代码将在https://github.com/Uddd821/StereoMV2D上提供。


### 论文摘要

Multi-view 3D object detection is a fundamental task in autonomous driving perception, where achieving a balance between detection accuracy and computational efficiency remains crucial. Sparse query-based 3D detectors efficiently aggregate object-relevant features from multi-view images through a set of learnable queries, offering a concise and end-to-end detection paradigm. Building on this foundation, MV2D leverages 2D detection results to provide high-quality object priors for query initialization, enabling higher precision and recall. However, the inherent depth ambiguity in single-frame 2D detections still limits the accuracy of 3D query generation. To address this issue, we propose StereoMV2D, a unified framework that integrates temporal stereo modeling into the 2D detection-guided multi-view 3D detector. By exploiting cross-temporal disparities of the same object across adjacent frames, StereoMV2D enhances depth perception and refines the query priors, while performing all computations efficiently within 2D regions of interest (RoIs). Furthermore, a dynamic confidence gating mechanism adaptively evaluates the reliability of temporal stereo cues through learning statistical patterns derived from the inter-frame matching matrix together with appearance consistency, ensuring robust detection under object appearance and occlusion. Extensive experiments on the nuScenes and Argoverse 2 datasets demonstrate that StereoMV2D achieves superior detection performance without incurring significant computational overhead. Code will be available at https://github.com/Uddd821/StereoMV2D.

---

## 26. SSCATeR: Sparse Scatter-Based Convolution Algorithm with Temporal Data Recycling for Real-Time 3D Object Detection in LiDAR Point Clouds

**论文链接:** [http://arxiv.org/abs/2512.08557v2](http://arxiv.org/abs/2512.08557v2)

**作者:** Alexander Dow, Manduhu Manduhu, Matheus Santos, Ben Bartlett, Gerard Dooly, James Riordan

**发布时间:** 2025-12-09

**备注:** 23 Pages, 27 Figures, This work has been submitted to the IEEE Sensors Journal for possible publication

### GPT解析

### 总结

该研究利用LiDAR扫描的连续运动特性，通过检测帧间变化的区域来集中目标检测，减少计算量同时保持准确性。

### 背景

LiDAR扫描会产生大量数据，传统方法需要处理整个点云，计算效率低下。

### 目的

减少处理时间，提高计算效率，同时保持检测准确性。

### 方法

使用滑动时间窗口和短步长处理，考虑时间维度存储卷积结果，忽略未变化区域，提出具有时间数据回收的稀疏散射卷积算法(SSCATeR)，仅对点云变化部分进行处理。

### 主要发现

处理时间减少了最多6.61倍，输出的特征图与传统稀疏卷积技术相同，大幅提高了网络计算效率。

### 结论

通过专注于变化区域并重用数据，该方法能够在保持相同检测精度的同时显著减少计算时间。

### 翻译

这项工作利用LiDAR扫描的连续扫描运动特性，通过将目标检测集中在从一帧到另一帧点数据发生变化的特定区域。我们通过使用短步长的滑动时间窗口实现这一点，并通过存储连续扫描间的卷积结果来考虑时间维度。这使得我们可以忽略未变化的区域，显著减少每前向传播的卷积操作数量，而不会牺牲准确性。这种数据重用方案为检测数据引入了极端稀疏性。为了利用这种稀疏性，我们扩展了之前基于散射的卷积工作，允许数据重用，并提出了具有时间数据回收的稀疏散射卷积算法(SSCATeR)。该操作将传入的LiDAR数据视为连续流，仅对点云的变化部分进行处理。通过这样做，我们实现了最多6.61倍的处理时间减少，同时获得相同的结果。我们的测试结果显示，我们方法输出的特征图与传统稀疏卷积技术产生的特征图相同，同时大大提高了网络的计算效率。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决LiDAR点云处理中的实时3D目标检测延迟问题。在自主无人机和自动驾驶车辆中，感知和检测（SAD）对于在动态环境中运行至关重要，而处理延迟会影响安全性。例如，商业无人机如DJI M300在200毫秒内可以移动超过4米，因此快速响应对于避免碰撞非常重要。现有方法即使在点云大部分区域不变的情况下，也会重复计算，导致不必要的延迟。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到LiDAR扫描具有时空特性，连续帧间大部分点云保持不变。他们思考如何将计算工作集中在最新变化的区域，避免重复计算。他们借鉴了PointPillars架构，扩展了自己之前关于基于散射的卷积的工作，并利用了稀疏卷积技术处理点云稀疏性。关键创新是引入短时间窗口（10毫秒）和变化地图机制，只处理变化区域并重用未变化区域的先前计算结果。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是利用LiDAR扫描的连续性，只处理点云中发生变化的部分，重用未变化区域的先前计算结果。整体流程包括：1)使用10毫秒间隔收集点云数据；2)在柱特征网络中将点云组织成柱，创建变化地图跟踪哪些区域发生变化；3)在卷积骨干中，只对变化站点应用稀疏散射卷积，使用反卷积移除先前输入的影响；4)在检测头生成3D边界框预测。这种方法通过'时间数据回收'机制显著减少计算量。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)时间数据回收机制，重用未变化区域的计算结果；2)短时间窗口处理（10毫秒而非传统的100毫秒）；3)变化地图技术，精确跟踪变化区域；4)稀疏散射卷积算法（SSCATeR）。相比之前工作，SSCATeR减少了59.85%的处理时间，避免了LSTM方法3.5-4倍的处理时间增加，避免了Transformer的慢推理问题，也避免了基于采样方法可能排除安全关键对象的风险，同时进一步利用了时间维度而非仅空间稀疏性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SSCATeR通过只处理LiDAR点云中的变化部分并重用未变化区域的先前计算结果，实现了实时3D目标检测，将处理时间减少了最多6.61倍而不牺牲准确性。'}


### 论文摘要

This work leverages the continuous sweeping motion of LiDAR scanning to concentrate object detection efforts on specific regions that receive a change in point data from one frame to another. We achieve this by using a sliding time window with short strides and consider the temporal dimension by storing convolution results between passes. This allows us to ignore unchanged regions, significantly reducing the number of convolution operations per forward pass without sacrificing accuracy. This data reuse scheme introduces extreme sparsity to detection data. To exploit this sparsity, we extend our previous work on scatter-based convolutions to allow for data reuse, and as such propose Sparse Scatter-Based Convolution Algorithm with Temporal Data Recycling (SSCATeR). This operation treats incoming LiDAR data as a continuous stream and acts only on the changing parts of the point cloud. By doing so, we achieve the same results with as much as a 6.61-fold reduction in processing time. Our test results show that the feature maps output by our method are identical to those produced by traditional sparse convolution techniques, whilst greatly increasing the computational efficiency of the network.

---

## 27. 论文ID: 2512.17879v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.17879v1.json'

---

## 28. Data-Driven Calibration of Large Liquid Detectors with Unsupervised Learning

**论文链接:** [http://arxiv.org/abs/2512.17866v1](http://arxiv.org/abs/2512.17866v1)

**作者:** Scott DeGraw, Steve Biller, Armin Reichold

**发布时间:** 2025-12-19

**备注:** 15 pages, 10 figures

### GPT解析

### 总结

本文提出了一种利用无监督深度学习从大型液体闪烁探测器的物理数据中提取光电倍增管校准时间常数的新方法。

### 背景

在大型液体闪烁探测器中，光电倍增管(PMT)的校准对于精确的物理测量至关重要，需要开发新的校准技术。

### 目的

开发一种能够从物理数据中自动提取PMT校准时间常数的方法，无需依赖传统的校准源或复杂的人工干预。

### 方法

使用无监督深度学习技术，在损失函数中嵌入简化的光子传输物理模型，将PMT校准常数作为自由参数，并假设单个事件代表点状发射，通过深度学习架构和自动微分框架实现可处理性。

### 主要发现

使用SNO+探测器的9300个PMT的数据，该方法成功为超过7500个在线PMT中的每一个提取了3个校准常数，利用的是放射性背景事件。

### 结论

这种基于深度学习的PMT校准方法可靠且有效，并且可以轻松推广到各种应用场景中，为大型探测器的校准提供了一种新的解决方案。

### 翻译

本文展示了一种新颖的方法，利用无监督深度学习的机制，从大型液体闪烁探测器的物理数据中提取光电倍增管(PMT)校准时间常数。该方法在损失函数中使用简化的光子传输物理模型，将PMT校准常数视为自由参数，并简单假设单个事件代表点状发射。因此，问题有效地简化为大规模回归问题，通过深度学习架构和自动微分框架使其变得可处理。使用SNO+探测器的9300个PMT的数据，该方法已被证明能够可靠地为超过7500个在线PMT中的每一个提取3个校准常数，使用的是放射性背景事件。我们相信这种方法可以轻松推广到广泛的应用中。


### 论文摘要

This paper demonstrates a novel method to extract photomultiplier tube (PMT) calibration timing constants in large liquid scintillation detectors from physics data using the machinery of unsupervised deep learning. The approach uses a simplified physical model of optical photon transport in the loss function, with PMT calibration constants treated as free parameters, and the simple assumption that individual events represent point-like emission. The problem is, thus, effectively reduced to that of regression on a very large scale, made tractable by deep learning architectures and automatic differentiation frameworks. Using data from the 9,300 PMTs in the SNO+ detector, the method has been shown to reliably extract 3 calibration constants for each of the over 7,500 online PMTs using radioactive background events. We believe that this basic approach can be straightforwardly generalised for a wide range of applications.

---

## 29. AdaptPrompt: Parameter-Efficient Adaptation of VLMs for Generalizable Deepfake Detection

**论文链接:** [http://arxiv.org/abs/2512.17730v1](http://arxiv.org/abs/2512.17730v1)

**作者:** Yichen Jiang, Mohammed Talha Alam, Sohail Ahmed Khan, Duc-Tien Dang-Nguyen, Fakhri Karray

**发布时间:** 2025-12-19

**备注:** Under Review

### GPT解析

### 总结

该研究利用大型视觉-语言模型CLIP解决深度伪造检测的泛化问题，提出Diff-Gen数据集和AdaptPrompt框架，在25个测试集上建立了新的最先进水平。

### 背景

图像生成技术的进步使得高度逼真的合成媒体广泛可用，增加了深度伪造检测的难度。主要挑战是泛化能力，因为针对有限类型生成器训练的检测器在面对未见过的模型时往往会失败。

### 目的

解决对可泛化深度伪造检测的迫切需求，利用大型视觉-语言模型识别各种生成技术合成的虚假内容。

### 方法

引入Diff-Gen数据集（包含10万张扩散生成伪造图像）和AdaptPrompt框架（参数高效的迁移学习方法，联合学习文本提示和视觉适配器，同时保持CLIP主干冻结）。通过层消融实验优化视觉编码器结构以增强高频生成伪影的保留。

### 主要发现

在Diff-Gen上训练的模型表现出更强的跨域泛化能力；在25个具有挑战性的测试集上评估，涵盖GAN、扩散模型和商业工具生成的合成内容；在标准和跨域场景中都建立了新的最先进水平；通过少样本泛化（仅320张图像）和源归因证明了框架的多功能性。

### 结论

通过利用大型视觉-语言模型和提出的新方法，显著提高了深度伪造检测的泛化能力，Diff-Gen数据集和AdaptPrompt框架为解决深度伪造检测中的泛化挑战提供了有效解决方案。

### 翻译

图像生成的最新进展使得高度逼真的合成媒体广泛可用，增加了可靠深度伪造检测的难度。关键挑战是泛化能力，因为针对有限类型生成器训练的检测器在面对未见过的模型时往往会失败。在这项工作中，我们通过利用大型视觉-语言模型（特别是CLIP）来识别各种生成技术合成的虚假内容，解决了对可泛化检测的迫切需求。首先，我们引入Diff-Gen，一个包含10万张扩散生成伪造图像的大规模基准数据集，捕获了传统GAN数据集所不具备的广泛频谱伪影。在Diff-Gen上训练的模型表现出更强的跨域泛化能力，特别是在面对未见过的图像生成器时。其次，我们提出AdaptPrompt，一个参数高效的迁移学习框架，联合学习任务特定的文本提示和视觉适配器，同时保持CLIP主干网络冻结。我们通过层消融实验进一步表明，剪裁视觉编码器的最后一个transformer块可以增强高频生成伪影的保留，显著提高检测准确率。我们的评估涵盖了25个具有挑战性的测试集，包括GAN、扩散模型和商业工具生成的合成内容，在标准和跨域场景中都建立了新的最先进水平。我们通过少样本泛化（使用仅320张图像）和源归因进一步证明了框架的多功能性，能够在封闭集设置中精确识别生成器架构。


### 论文摘要

Recent advances in image generation have led to the widespread availability of highly realistic synthetic media, increasing the difficulty of reliable deepfake detection. A key challenge is generalization, as detectors trained on a narrow class of generators often fail when confronted with unseen models. In this work, we address the pressing need for generalizable detection by leveraging large vision-language models, specifically CLIP, to identify synthetic content across diverse generative techniques. First, we introduce Diff-Gen, a large-scale benchmark dataset comprising 100k diffusion-generated fakes that capture broad spectral artifacts unlike traditional GAN datasets. Models trained on Diff-Gen demonstrate stronger cross-domain generalization, particularly on previously unseen image generators. Second, we propose AdaptPrompt, a parameter-efficient transfer learning framework that jointly learns task-specific textual prompts and visual adapters while keeping the CLIP backbone frozen. We further show via layer ablation that pruning the final transformer block of the vision encoder enhances the retention of high-frequency generative artifacts, significantly boosting detection accuracy. Our evaluation spans 25 challenging test sets, covering synthetic content generated by GANs, diffusion models, and commercial tools, establishing a new state-of-the-art in both standard and cross-domain scenarios. We further demonstrate the framework's versatility through few-shot generalization (using as few as 320 images) and source attribution, enabling the precise identification of generator architectures in closed-set settings.

---

## 30. HydroGym: A Reinforcement Learning Platform for Fluid Dynamics

**论文链接:** [http://arxiv.org/abs/2512.17534v1](http://arxiv.org/abs/2512.17534v1)

**作者:** Christian Lagemann, Sajeda Mokbel, Miro Gondrum, Mario Rüttgers, Jared Callaham, Ludger Paehler, Samuel Ahnert, Nicholas Zolman, Kai Lagemann, Nikolaus Adams, Matthias Meinke, Wolfgang Schröder, Jean-Christophe Loiseau, Esther Lagemann, Steven L. Brunton

**发布时间:** 2025-12-19

### GPT解析

### 总结

本研究介绍了HydroGym，一个用于流体控制研究的独立求解器的强化学习平台，集成了复杂的流动控制基准测试、可扩展运行时基础设施和最先进的强化学习算法，包含42个经过验证的环境。

### 背景

流体流动建模和控制对交通、能源和医学等科学工程领域至关重要，可带来升力增加、阻力减少等效益。然而流体控制面临高维、非线性、多尺度等挑战，强化学习应用受限于缺乏标准化基准平台和高计算需求。

### 目的

解决流体控制中强化学习应用的挑战，提供标准化平台和降低计算需求。

### 方法

开发HydroGym平台，包含42个验证环境（从层流到湍流），提供不可微和可微求解器，通过梯度增强优化提高样本效率。

### 主要发现

强化学习代理能发现边界层操纵等稳健控制原理；迁移学习显示控制器适应新条件可减少50%训练需求；平台具有高度可扩展性。

### 结论

HydroGym为流体控制研究提供了全面解决方案，促进流体动力学、机器学习和控制领域的交叉研究，推动科技进步。

### 翻译

建模和控制流体流动对于科学和工程的几个领域（包括交通、能源和医学）至关重要。有效的流动控制可以带来，例如，升力增加、阻力减少、混合增强和噪声减少。然而，控制流体面临几个重大挑战，包括空间和时间上的高维、非线性和多尺度相互作用。强化学习（RL）最近在复杂领域（如机器人和蛋白质折叠）中显示出巨大成功，但其在流动控制中的应用受到缺乏标准化基准平台和流体模拟计算需求的阻碍。为了解决这些挑战，我们介绍了HydroGym，这是一个用于流动控制研究的独立求解器的强化学习平台。HydroGym集成了复杂的流动控制基准测试、可扩展的运行时基础设施和最先进的强化学习算法。我们的平台包括42个经过验证的环境，涵盖从经典层流到复杂的三维湍流场景，在广泛的雷诺数范围内进行了验证。我们为传统强化学习提供了不可微求解器，为通过梯度增强优化显著提高样本效率的可微求解器。全面的评估显示，强化学习代理在各种配置中一致地发现稳健的控制原理，如边界层操纵、声学反馈干扰和尾流重组。迁移学习研究表明，在一个雷诺数或几何形状下学习的控制器能够有效地适应新条件，需要大约50%更少的训练周期。HydroGym平台具有高度可扩展性和可扩展性，为流体动力学、机器学习和控制领域的研究人员提供了一个框架，用于添加环境、代理模型和控制算法，以推进科学和技术。


### 论文摘要

Modeling and controlling fluid flows is critical for several fields of science and engineering, including transportation, energy, and medicine. Effective flow control can lead to, e.g., lift increase, drag reduction, mixing enhancement, and noise reduction. However, controlling a fluid faces several significant challenges, including high-dimensional, nonlinear, and multiscale interactions in space and time. Reinforcement learning (RL) has recently shown great success in complex domains, such as robotics and protein folding, but its application to flow control is hindered by a lack of standardized benchmark platforms and the computational demands of fluid simulations. To address these challenges, we introduce HydroGym, a solver-independent RL platform for flow control research. HydroGym integrates sophisticated flow control benchmarks, scalable runtime infrastructure, and state-of-the-art RL algorithms. Our platform includes 42 validated environments spanning from canonical laminar flows to complex three-dimensional turbulent scenarios, validated over a wide range of Reynolds numbers. We provide non-differentiable solvers for traditional RL and differentiable solvers that dramatically improve sample efficiency through gradient-enhanced optimization. Comprehensive evaluation reveals that RL agents consistently discover robust control principles across configurations, such as boundary layer manipulation, acoustic feedback disruption, and wake reorganization. Transfer learning studies demonstrate that controllers learned at one Reynolds number or geometry adapt efficiently to new conditions, requiring approximately 50% fewer training episodes. The HydroGym platform is highly extensible and scalable, providing a framework for researchers in fluid dynamics, machine learning, and control to add environments, surrogate models, and control algorithms to advance science and technology.

---

## 31. UCoder: Unsupervised Code Generation by Internal Probing of Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.17385v1](http://arxiv.org/abs/2512.17385v1)

**作者:** Jiajun Wu, Jian Yang, Wei Zhang, Lin Jing, Yuqing Ma, Ensheng Shi, Yuchi Ma, Zhoujun Li, Xianglong Liu

**发布时间:** 2025-12-19

### GPT解析

### 总结

本文提出了一种名为IPC的无监督框架，通过探测大型语言模型内部知识和置信度模式，实现了无需外部语料库的代码生成，性能可与监督方法媲美且减少了对标注数据和计算资源的依赖。

### 背景

大型语言模型在代码生成任务中表现出色，但其有效性依赖于有监督训练，需要大量标注或未标注数据，而这些数据往往难以大规模获取且成本高昂。

### 目的

提出一种无需任何外部语料库（包括未标注代码片段）的无监督框架，用于代码生成任务，减少对标注数据的依赖。

### 方法

IPC框架包含问题空间探测、测试理解探测、解决方案空间探测、知识巩固和强化，用于探测LLMs内部知识和置信度模式；通过自一致性机制和基于表示的质量识别可靠代码候选，训练UCoder模型。

### 主要发现

在多个代码基准测试上验证，无监督方法可与监督方法性能相当；内部模型状态包含关于代码质量和正确性的丰富信号，适当利用这些信号可实现有效的无监督学习。

### 结论

为在资源受限场景下训练代码大型语言模型开辟了新方向，证明了无监督方法在代码生成任务中的潜力。

### 翻译

大型语言模型在代码生成任务中表现出了卓越的能力。然而，它们的有效性很大程度上依赖于使用大量标注（如问答对）或未标注数据集（如代码片段）进行监督训练，而这些数据通常难以大规模获取且成本高昂。为解决这一局限性，本文提出了一种名为IPC的方法，这是一个无监督框架，利用大型语言模型内部探测进行代码生成，无需任何外部语料库，甚至是未标注的代码片段。我们引入了问题空间探测、测试理解探测、解决方案空间探测以及知识巩固和强化，来探测LLMs内部存在的知识和置信度模式。此外，IPC通过自一致性机制和基于表示的质量估计来识别可靠的代码候选，以训练UCoder（具有无监督学习的编码器）。我们在多个代码基准测试上验证了所提出的方法，证明无监督方法可以与监督方法实现竞争性性能，同时显著减少对标注数据和计算资源的依赖。分析实验表明，内部模型状态包含关于代码质量和正确性的丰富信号，适当利用这些信号能够实现代码生成任务的有效无监督学习，为在资源受限场景下训练代码大型语言模型开辟了新方向。


### 论文摘要

Large language models (LLMs) have demonstrated remarkable capabilities in code generation tasks. However, their effectiveness heavily relies on supervised training with extensive labeled (e.g., question-answering pairs) or unlabeled datasets (e.g., code snippets), which are often expensive and difficult to obtain at scale. To address this limitation, this paper introduces a method IPC, an unsupervised framework that leverages Internal Probing of LLMs for Code generation without any external corpus, even unlabeled code snippets. We introduce the problem space probing, test understanding probing, solution space probing, and knowledge consolidation and reinforcement to probe the internal knowledge and confidence patterns existing in LLMs. Further, IPC identifies reliable code candidates through self-consistency mechanisms and representation-based quality estimation to train UCoder (coder with unsupervised learning). We validate the proposed approach across multiple code benchmarks, demonstrating that unsupervised methods can achieve competitive performance compared to supervised approaches while significantly reducing the dependency on labeled data and computational resources. Analytic experiments reveal that internal model states contain rich signals about code quality and correctness, and that properly harnessing these signals enables effective unsupervised learning for code generation tasks, opening new directions for training code LLMs in resource-constrained scenarios.

---

## 32. An Interpretable Latent Space reveals changing dynamics of European heatwaves

**论文链接:** [http://arxiv.org/abs/2512.17097v1](http://arxiv.org/abs/2512.17097v1)

**作者:** Tamara Happé, Jasper Wijnands, Paolo Scussolini, Peter Pfleiderer, Dim Coumou

**发布时间:** 2025-12-18

**备注:** 16 pages, 7 figures, 3 appendix figures, submitted for peer review to Machine Learning: Earth

### GPT解析

### 总结

该研究利用深度学习技术基于大气环流对欧洲热浪进行分类和时间变化研究，发现不同类型热浪呈现不同变化趋势，强调了单独研究每种热浪类型的必要性。

### 背景

由于气候变化，热浪变得越来越频繁和强烈，西欧经历了北半球中纬度地区最强的热浪趋势。部分温度趋势是由环流变化引起的，但这些变化在气候模型中没有准确捕捉。

### 目的

使用深度学习技术基于大气环流对欧洲热浪进行分类，并研究这些热浪相关的时间变化。

### 方法

使用变分自编码器（VAE）降低热浪样本的维度，在提取的特征上对热浪进行聚类。VAE在大型集合气候模型模拟上进行训练，并能很好地泛化到观测数据。研究引入了新的可解释性方法来研究潜在空间。

### 主要发现

ERA5中与热浪相关的环流特征与气候模型热浪一致；大西洋羽流型热浪随时间变得越来越频繁，而大西洋高压型热浪变得越来越少；每种热浪类型都在经历其独特的环流变化，例如大西洋低压型热浪显示出随时间大西洋沿岸低压系统加深的趋势。

### 结论

需要分别研究每种热浪类型，突显了单独研究的必要性；该方法可用于增强极端事件的特定方面；如果当前趋势持续，热浪环流可能会在未来发生变化，在某些情况下特征会加强。

### 翻译

由于气候变化，热浪变得越来越频繁和强烈，西欧经历了北半球中纬度地区最强的热浪趋势。部分温度趋势是由环流变化引起的，这些变化在气候模型中没有准确捕捉。我们在此部署深度学习技术，基于大气环流对欧洲热浪进行分类，并研究它们相关的时间变化。我们使用变分自编码器（VAE）降低热浪样本的维度，然后在提取的特征上对它们进行聚类。VAE在大型集合气候模型模拟上进行训练，我们展示VAE无需迁移学习就能很好地泛化到ERA5再分析中的观测热浪环流。ERA5中与热浪相关的环流特征与气候模型热浪一致。回归分析显示，大西洋羽流型热浪随时间变得越来越频繁，而大西洋高压型热浪变得越来越少。我们引入了新的简单可解释性方法来研究潜在空间，包括特征重要性的识别和时间变化。我们研究与潜在空间中最重要的节点相关的环流特征，以及潜在空间如何随时间变化。例如，我们发现大西洋低压型热浪显示出随时间大西洋沿岸低压系统加深的趋势。每种热浪类型都在经历其独特的环流变化，突显了分别研究每种热浪类型的必要性。我们的方法还可用于增强极端事件的特定方面，我们说明了如果当前趋势持续，热浪环流在未来可能如何变化，在某些情况下特征会加强。


### 论文摘要

Due to climate change, heatwaves are becoming more frequent and intense, with western Europe experiencing the strongest trends in the Northern Hemisphere mid-latitudes. Part of the temperature trends are caused by circulation changes, which are not accurately captured in climate models. Here we deploy Deep Learning techniques to classify European heatwaves based on their atmospheric circulation and to study their associated changes over time. We use a Variational Autoencoder (VAE) to reduce the dimensionality of the heatwave samples, after which we cluster them on their extraced features. The VAE is trained on large ensemble climate model simulations and we show that the VAE generalizes well to observed heatwave circulations in ERA5 reanalysis, without the need for transfer learning. The circulation features relevant for heatwaves in ERA5 are consistent with the climate model heatwaves. Regression analysis reveals that the Atlantic Plume type of heatwaves are becoming more frequent over time, while the Atlantic High heatwaves are becoming less frequent. We introduce new and straightforward interpretability methods to study the latent space, including feature importance identification and changes over time. We investigate which circulation features are associated with the most important nodes in the latent space and how the latent space changes over time. For example, we find that the Atlantic Low heatwave shows a deepening of the low pressure system off the Atlantic coast over time. Each heatwave type is undergoing unique changes in their circulation, highlighting the necessity to study each heatwave type separately. Our method can furthermore be used to boost specific aspects of extreme events, and we illustrate how heatwave circulation could change in the future if the current trends persist, with in some cases an intensification of features.

---

## 33. 论文ID: 2512.17541v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.17541v1.json'

---

## 34. 论文ID: 2512.17371v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.17371v1.json'

---

## 35. A robust morphological classification method for galaxies using dual-encoding contrastive learning and multi-clustering voting on JWST/NIRCam images

**论文链接:** [http://arxiv.org/abs/2512.17162v1](http://arxiv.org/abs/2512.17162v1)

**作者:** Xiaolei Yin, Guanwen Fang, Shiying Lu, Zesen Lin, Yao Dai, Chichun Zhou

**发布时间:** 2025-12-19

**备注:** Published in A&A

### GPT解析

### 总结

本研究介绍了一个名为USmorph的两步星系形态学分类框架，结合了无监督和有监督机器学习方法，对COSMOS-Web场中的46,176个星系进行了分类，并验证了该分类系统的可靠性。

### 背景

星系形态分类是天文学研究中的重要内容，随着观测技术的进步，需要更有效的分类方法来处理大量星系数据。

### 目的

开发并验证一个可靠的星系形态分类系统，用于即将进行的中国空间站望远镜大天区巡天研究。

### 方法

采用双步框架：增强无监督机器学习步骤，使用双编码器架构(ConvNeXt和ViT)编码图像，应用对比学习提取特征，使用主成分分析降维；然后使用改进的框架对星系进行形态分类，并通过参数化和非参数化测量验证分类结果。

### 主要发现

成功将46,176个星系分为五类：33%球形(SPH)、25%早期盘状星系(ETD)、25%晚期盘状星系(LTD)、7%不规则星系(IRR)和10%未分类星系(UNC)；形态参数分析显示，SPH和ETD星系具有较高的Sérsic指数、Gini系数和集中度，倾向于更以凸起为主导且更紧凑。

### 结论

USmorph分类系统可靠有效，能够准确区分不同形态的星系，适用于未来的大天区星系巡天研究。

### 翻译

USmorph两步星系形态分类框架成功结合了无监督机器学习与有监督机器学习方法。为增强无监督学习步骤，我们采用双编码器架构(ConvNeXt和ViT)有效编码图像，使用对比学习准确提取特征，并应用主成分分析高效降维。基于此改进框架，使用JWST近红外图像对COSMOS-Web场中选取的46,176个红移范围0<z<4.2的星系分为五类：33%球形(SPH)、25%早期盘状星系(ETD)、25%晚期盘状星系(LTD)、7%不规则星系(IRR)和10%未分类星系(UNC)。我们还对大质量星系(M*>10^9 M⊙)进行了参数化(Sérsic指数n和有效半径re)和非参数化测量(Gini系数G、光二阶矩M20、集中度C、多重性Ψ以及MID统计中的其他三个参数)，以验证星系形态分类系统的有效性。形态参数分析表明，与其它类型星系相比，SPH和ETD星系具有较高的n、G和C值，倾向于更以凸起为主导且更紧凑。这证明了该分类系统的可靠性，该系统将对中国空间站望远镜即将进行的大天区巡天研究有用。


### 论文摘要

The two-step galaxy morphology classification framework {\tt USmorph} successfully combines unsupervised machine learning (UML) with supervised machine learning (SML) methods. To enhance the UML step, we employed a dual-encoder architecture (ConvNeXt and ViT) to effectively encode images, contrastive learning to accurately extract features, and principal component analysis to efficiently reduce dimensionality. Based on this improved framework, a sample of 46,176 galaxies at $0<z<4.2$, selected in the COSMOS-Web field, is classified into five types using the JWST near-infrared images: 33\% spherical (SPH), 25\% early-type disk (ETD), 25\% late-type disk (LTD), 7\% irregular (IRR), and 10\% unclassified (UNC) galaxies. We also performed parametric (S{é}rsic index, $n$,and effective radius, $r_{\rm e}$) and nonparametric measurements (Gini coefficient, $G$, the second-order moment of light, $M_{\rm 20}$, concentration, $C$, multiplicity, $Ψ$, and three other parameters from the MID statistics) for massive galaxies ($M_*>10^9 M_\odot$) to verify the validity of our galaxy morphological classification system. The analysis of morphological parameters is consistent with our classification system: SPH and ETD galaxies with higher $n$, $G$, and $C$ tend to be more bulge-dominated and more compact compared with other types of galaxies. This demonstrates the reliability of this classification system, which will be useful for a forthcoming large-sky survey from the Chinese Space Station Telescope.

---

## 36. Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation

**论文链接:** [http://arxiv.org/abs/2512.17762v1](http://arxiv.org/abs/2512.17762v1)

**作者:** Luca Miglior, Matteo Tolloso, Alessio Gravina, Davide Bacciu

**发布时间:** 2025-12-19

### GPT解析

### 总结

本文介绍了ECHO基准测试，专门用于评估图神经网络处理长距离图传播的能力，包含三个合成图任务和两个真实世界数据集，通过测试揭示了现有GNN架构在长距离传播方面的性能差距。

### 背景

在图神经网络研究中，有效捕获长距离交互仍然是一个基本但尚未解决的关键挑战，这对科学不同领域的应用至关重要。

### 目的

为了系统地解决长距离交互捕获问题，引入ECHO基准测试，旨在严格评估GNN在处理非常长范围图传播方面的能力。

### 方法

ECHO包括三个合成图任务（单源最短路径、节点离心率和图直径），构建在多样化和结构上具有挑战性的拓扑上，引入显著信息瓶颈；还包括两个真实世界数据集：ECHO-Charge（预测原子部分电荷）和ECHO-Energy（预测分子总能量），参考计算基于密度泛函理论水平。

### 主要发现

通过对流行GNN架构的广泛基准测试，发现了明显的性能差距，强调了真实长范围传播的难度，并突出了能够克服固有局限性的设计选择。

### 结论

ECHO为评估长距离信息传播设定了新的标准，同时也为科学中人工智能的需求提供了引人注目的例证。

### 翻译

在图神经网络研究中，有效捕获长距离交互仍然是一个基本但尚未解决的关键挑战，对科学不同领域的应用至关重要。为了系统地解决这个问题，我们引入了ECHO（评估长跳通信能力），这是一个新的基准测试，专门设计用于严格评估GNN处理非常长范围图传播的能力。ECHO包括三个合成图任务，即单源最短路径、节点离心率和图直径，每个任务都构建在多样化和结构上具有挑战性的拓扑上，专门设计以引入显著的信息瓶颈。ECHO还包括两个真实世界数据集：ECHO-Charge和ECHO-Energy，它们分别为预测原子部分电荷和分子总能量定义了基于化学的基准，参考计算在密度泛函理论水平上获得。这两个任务本质上都依赖于捕获复杂的长距离分子相互作用。我们对流行GNN架构的广泛基准测试揭示了明显的性能差距，强调了真实长范围传播的难度，并突出了能够克服固有局限性的设计选择。ECHO因此为评估长距离信息传播设定了新的标准，同时也为科学中人工智能的需求提供了一个引人注目的例证。


### 论文摘要

Effectively capturing long-range interactions remains a fundamental yet unresolved challenge in graph neural network (GNN) research, critical for applications across diverse fields of science. To systematically address this, we introduce ECHO (Evaluating Communication over long HOps), a novel benchmark specifically designed to rigorously assess the capabilities of GNNs in handling very long-range graph propagation. ECHO includes three synthetic graph tasks, namely single-source shortest paths, node eccentricity, and graph diameter, each constructed over diverse and structurally challenging topologies intentionally designed to introduce significant information bottlenecks. ECHO also includes two real-world datasets, ECHO-Charge and ECHO-Energy, which define chemically grounded benchmarks for predicting atomic partial charges and molecular total energies, respectively, with reference computations obtained at the density functional theory (DFT) level. Both tasks inherently depend on capturing complex long-range molecular interactions. Our extensive benchmarking of popular GNN architectures reveals clear performance gaps, emphasizing the difficulty of true long-range propagation and highlighting design choices capable of overcoming inherent limitations. ECHO thereby sets a new standard for evaluating long-range information propagation, also providing a compelling example for its need in AI for science.

---

## 37. Spatially-informed transformers: Injecting geostatistical covariance biases into self-attention for spatio-temporal forecasting

**论文链接:** [http://arxiv.org/abs/2512.17696v1](http://arxiv.org/abs/2512.17696v1)

**作者:** Yuri Calleo

**发布时间:** 2025-12-19

### GPT解析

### 总结

该研究提出了一种空间感知Transformer混合架构，将地统计学归纳偏差直接注入自注意力机制，实现了高性能的时空过程建模，同时保持了物理感知和数据驱动学习的优势。

### 背景

高维时空过程建模面临经典地统计学与深度学习之间的根本分歧。高斯过程提供理论一致性和精确不确定性量化，但计算复杂度高，对大规模传感器网络不实用；现代Transformer架构擅长序列建模但缺乏几何归纳偏差，将空间传感器视为排列不变标记，没有距离的本理解。

### 目的

开发一种能够结合地统计学理论严谨性与深度学习灵活性的混合架构，用于高维时空过程建模。

### 方法

提出空间感知Transformer，通过可学习协方差核将地统计学归纳偏差注入自注意力机制；将注意力结构形式化分解为平稳物理先验和非平稳数据驱动残差，施加软拓扑约束，有利于空间近程交互同时保留建模复杂动力学的能力。

### 主要发现

实现了'深度变异学'现象，网络通过反向传播成功恢复底层过程的真实空间衰减参数；在合成高斯随机场和真实世界交通基准上的实验证实，该方法优于最先进的图神经网络；提供校准良好的概率预测，兼具卓越的预测准确性和可靠性。

### 结论

该方法有效地弥合了物理感知建模与数据驱动学习之间的差距，为高维时空过程建模提供了新的解决方案。

### 翻译

高维时空过程的建模呈现出经典地统计学的概率严谨性与深度学习的灵活、高容量表示之间的根本分歧。虽然高斯过程提供理论一致性和精确的不确定性量化，但其 prohibitive 计算扩展使其对大规模传感器网络不切实际。相反，现代Transformer架构擅长序列建模，但固有地缺乏几何归纳偏差，将空间传感器视为排列不变的标记，没有对距离的本理解。在这项工作中，我们提出了一种空间感知Transformer，一种混合架构，通过可学习的协方差核将地统计学归纳偏差直接注入自注意力机制。通过将注意力结构形式化分解为平稳物理先验和非平稳数据驱动残差，我们施加了一个软拓扑约束，有利于空间近程交互，同时保留建模复杂动力学的能力。我们证明了'深度变异学'现象，其中网络通过反向传播成功恢复底层过程的真实空间衰减参数。在合成高斯随机场和真实世界交通基准上的广泛实验证实，我们的方法优于最先进的图神经网络。此外，严格的统计验证证实，所提出的方法不仅提供卓越的预测准确性，还提供校准良好的概率预测，有效地弥合了物理感知建模与数据驱动学习之间的差距。


### 论文摘要

The modeling of high-dimensional spatio-temporal processes presents a fundamental dichotomy between the probabilistic rigor of classical geostatistics and the flexible, high-capacity representations of deep learning. While Gaussian processes offer theoretical consistency and exact uncertainty quantification, their prohibitive computational scaling renders them impractical for massive sensor networks. Conversely, modern transformer architectures excel at sequence modeling but inherently lack a geometric inductive bias, treating spatial sensors as permutation-invariant tokens without a native understanding of distance. In this work, we propose a spatially-informed transformer, a hybrid architecture that injects a geostatistical inductive bias directly into the self-attention mechanism via a learnable covariance kernel. By formally decomposing the attention structure into a stationary physical prior and a non-stationary data-driven residual, we impose a soft topological constraint that favors spatially proximal interactions while retaining the capacity to model complex dynamics. We demonstrate the phenomenon of ``Deep Variography'', where the network successfully recovers the true spatial decay parameters of the underlying process end-to-end via backpropagation. Extensive experiments on synthetic Gaussian random fields and real-world traffic benchmarks confirm that our method outperforms state-of-the-art graph neural networks. Furthermore, rigorous statistical validation confirms that the proposed method delivers not only superior predictive accuracy but also well-calibrated probabilistic forecasts, effectively bridging the gap between physics-aware modeling and data-driven learning.

---

## 38. 论文ID: 2512.17453v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.17453v1.json'

---

## 39. 论文ID: 2512.17352v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.17352v1.json'

---

## 40. From Priors to Predictions: Explaining and Visualizing Human Reasoning in a Graph Neural Network Framework

**论文链接:** [http://arxiv.org/abs/2512.17255v1](http://arxiv.org/abs/2512.17255v1)

**作者:** Quan Do, Caroline Ahn, Leah Bakst, Michael Pascale, Joseph T. McGuire, Chantal E. Stern, Michael E. Hasselmo

**发布时间:** 2025-12-19

**备注:** 44 pages, 7 figures, 3 suppl figures

### GPT解析

### 总结

这项研究引入了一个结合图论和图神经网络的框架，将归纳偏置形式化为结构和抽象上的显式先验，用于理解人类推理和开发更符合人类的AI系统。

### 背景

人类能够通过归纳偏置解决新的推理问题，但这些偏置的计算形式和神经实现仍不清楚。

### 目的

形式化归纳偏置为显式、可操作的先验，解释人类推理中的个体差异，并开发更符合人类的AI系统。

### 方法

结合图论和图神经网络创建框架，使用改编自ARC的人类行为数据集，开发优化管道搜索图配置，以及可视化方法识别关键计算图。

### 主要发现

基于图的先验可以解释人类解决方案中的个体差异；泛化依赖于特定的先验结构和内部处理；不正确或不完整的先验会导致类似人类的错误。

### 结论

该研究为建模泛化背后的表征假设和计算动力学提供了有原则的、可解释的框架，为人类推理提供了新见解，并为更符合人类的AI系统奠定了基础。

### 翻译

人类擅长通过归纳偏置解决新的推理问题，这些偏置是对哪些实体和关系重要的假设。然而，这些偏置的计算形式和神经实现仍不清楚。我们引入了一个结合图论和图神经网络的框架，将归纳偏置形式化为结构和抽象上的显式、可操作的先验。使用改编自抽象与推理语料库的人类行为数据集，我们表明基于图的先验差异可以解释人类解决方案中的个体差异。我们的方法包括一个优化管道，搜索图配置，变化边缘连通性和节点抽象，以及一种识别计算图的可视化方法，即对模型预测最重要的节点和边的子集。系统性消融研究揭示了泛化如何依赖于特定的先验结构和内部处理，暴露了为什么类似人类的错误会从不正确或不完整的先验中出现。这项工作为建模泛化背后的表征假设和计算动力学提供了一个有原则的、可解释的框架，为人类推理提供了新的见解，并为更符合人类的AI系统奠定了基础。


### 论文摘要

Humans excel at solving novel reasoning problems from minimal exposure, guided by inductive biases, assumptions about which entities and relationships matter. Yet the computational form of these biases and their neural implementation remain poorly understood. We introduce a framework that combines Graph Theory and Graph Neural Networks (GNNs) to formalize inductive biases as explicit, manipulable priors over structure and abstraction. Using a human behavioral dataset adapted from the Abstraction and Reasoning Corpus (ARC), we show that differences in graph-based priors can explain individual differences in human solutions. Our method includes an optimization pipeline that searches over graph configurations, varying edge connectivity and node abstraction, and a visualization approach that identifies the computational graph, the subset of nodes and edges most critical to a model's prediction. Systematic ablation reveals how generalization depends on specific prior structures and internal processing, exposing why human like errors emerge from incorrect or incomplete priors. This work provides a principled, interpretable framework for modeling the representational assumptions and computational dynamics underlying generalization, offering new insights into human reasoning and a foundation for more human aligned AI systems.

---

## 41. Systemic Risk Radar: A Multi-Layer Graph Framework for Early Market Crash Warning

**论文链接:** [http://arxiv.org/abs/2512.17185v1](http://arxiv.org/abs/2512.17185v1)

**作者:** Sandeep Neela

**发布时间:** 2025-12-19

**备注:** Preprint

### GPT解析

### 总结

本研究提出了系统风险雷达(SRR)框架，通过将金融市场建模为多层图来检测系统脆弱性和崩溃状态的早期预警信号，并在三大危机中验证了其有效性。

### 背景

金融危机在各部门、市场和投资者行为中累积结构性脆弱性时出现，预测这些系统转变具有挑战性，因为它们源于市场参与者之间不断变化的互动，而非孤立的价格变动。

### 目的

开发一个能够检测金融系统脆弱性和崩溃状态转变早期迹象的框架，以提前预警金融危机。

### 方法

提出系统风险雷达(SRR)框架，将金融市场建模为多层图，并在互联网泡沫崩溃、全球金融危机和COVID-19冲击三大危机中进行评估，比较了快照图神经网络、时序GNN原型以及逻辑回归和随机森林等基线模型。

### 主要发现

结构网络信息比仅基于特征的模型提供了更有用的早期预警信号，图派生特征能够有效捕获压力事件期间市场结构的有意义变化。

### 结论

研究结果支持扩展SRR框架，增加额外的图层次（如行业/因子敞口、情绪）和更复杂的时序架构（如LSTM/GRU或Transformer编码器），以更好地处理多样化的危机类型。

### 翻译

金融危机在各部门、市场和投资者行为中累积结构性脆弱性时出现。预测这些系统转变具有挑战性，因为它们源于市场参与者之间不断变化的互动，而不仅仅是孤立的价格变动。我们提出了'系统风险雷达'(SRR)框架，该框架将金融市场建模为多层图，以检测系统脆弱性和崩溃状态转变的早期迹象。我们在三大危机中评估了SRR：互联网泡沫崩溃、全球金融危机和COVID-19冲击。我们的实验比较了快照图神经网络、简化的时序GNN原型以及标准基线（逻辑回归和随机森林）。结果表明，与仅基于特征的模型相比，结构网络信息提供了有用的早期预警信号。SRR的这种基于相关性的实例表明，图派生特征能够捕获压力事件期间市场结构的有意义变化。这些发现促使我们扩展SRR，增加额外的图层次（行业/因子敞口、情绪）和更具表现力的时序架构（LSTM/GRU或Transformer编码器），以更好地处理不同类型的危机。


### 论文摘要

Financial crises emerge when structural vulnerabilities accumulate across sectors, markets, and investor behavior. Predicting these systemic transitions is challenging because they arise from evolving interactions between market participants, not isolated price movements alone. We present Systemic Risk Radar (SRR), a framework that models financial markets as multi-layer graphs to detect early signs of systemic fragility and crash-regime transitions.   We evaluate SRR across three major crises: the Dot-com crash, the Global Financial Crisis, and the COVID-19 shock. Our experiments compare snapshot GNNs, a simplified temporal GNN prototype, and standard baselines (logistic regression and Random Forest). Results show that structural network information provides useful early-warning signals compared to feature-based models alone.   This correlation-based instantiation of SRR demonstrates that graph-derived features capture meaningful changes in market structure during stress events. The findings motivate extending SRR with additional graph layers (sector/factor exposure, sentiment) and more expressive temporal architectures (LSTM/GRU or Transformer encoders) to better handle diverse crisis types.

---

## 42. 论文ID: 2512.17129v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.17129v1.json'

---

## 43. 论文ID: 2512.17084v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.17084v1.json'

---

## 44. InsertAnywhere: Bridging 4D Scene Geometry and Diffusion Models for Realistic Video Object Insertion

**论文链接:** [http://arxiv.org/abs/2512.17504v1](http://arxiv.org/abs/2512.17504v1)

**作者:** Hoiyeong Jin, Hyojin Jang, Jeongho Kim, Junha Hyung, Kinam Kim, Dongjin Kim, Huijin Choi, Hyeonji Kim, Jaegul Choo

**发布时间:** 2025-12-19

**备注:** 16 pages, project page: https://myyzzzoooo.github.io/InsertAnywhere/

### GPT解析

### 总结

该研究提出了InsertAnywhere框架，实现了几何一致的对象放置和外观忠实于原始视频的合成，解决了视频对象插入中的4D场景理解、遮挡处理和光照效果等关键挑战。

### 背景

基于扩散的视频生成技术为可控视频编辑开辟了新可能性，但现实中的视频对象插入(VOI)仍面临挑战，主要由于有限的4D场景理解以及对遮挡和光照效果的不当处理。

### 目的

开发一个能够实现几何一致对象放置和外观忠实视频合成的视频对象插入框架，解决现有方法在4D场景理解、遮挡处理和光照效果方面的不足。

### 方法

1. 提出了4D感知的掩码生成模块，重建场景几何并保持时间一致性和遮挡一致性；2. 扩展了基于扩散的视频生成模型，共同合成插入对象及其周围局部变化；3. 构建了ROSE++光照感知合成数据集，支持监督训练。

### 主要发现

该框架能够在各种真实世界场景中产生几何合理且视觉连贯的对象插入效果，显著优于现有研究和商业模型。

### 结论

InsertAnywhere框架通过结合几何一致的对象放置和外观忠实于原始视频的合成，为可控视频编辑提供了新的有效解决方案。

### 翻译

基于扩散的视频生成最新进展为可控视频编辑开辟了新的可能性，但由于有限的4D场景理解以及对遮挡和光照效果的不当处理，现实中的视频对象插入(VOI)仍然具有挑战性。我们提出了InsertAnywhere，一种新的VOI框架，实现了几何一致的对象放置和外观忠实于原始视频的合成。我们的方法从4D感知的掩码生成模块开始，该模块重建场景几何并在保持时间一致性和遮挡一致性的同时传播用户指定的对象放置。在此基础上，我们扩展了基于扩散的视频生成模型，以共同合成插入对象及其周围局部变化，如光照和阴影。为了支持监督训练，我们引入了ROSE++，这是一个光照感知的合成数据集，通过将ROSE对象移除数据集转换为对象移除视频、对象存在视频和VLM生成的参考图像三元组来构建。通过大量实验，我们证明我们的框架能够在各种真实世界场景中产生几何合理且视觉连贯的对象插入效果，显著优于现有研究和商业模型。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视频对象插入(VOI)的问题，具体是现有方法在4D场景理解和处理遮挡、光照效果方面的不足。这个问题在商业广告、电影后期制作和虚拟产品植入等领域非常重要，因为这些应用需要高质量的对象插入技术，而现有方法难以在复杂场景中实现几何一致且视觉效果真实的对象插入。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将VOI问题分解为两个主要挑战：4D感知对象放置和高保真视频生成。他们设计了一个两阶段框架：第一阶段使用4D场景重建和用户交互式放置生成几何一致的掩码序列；第二阶段基于扩散模型生成高质量视频。作者借鉴了Uni4D的4D场景建模、SEA-RAFT的光流计算、SAM2的分割技术，并基于ROSE数据集构建了新的ROSE++数据集，同时使用LoRA技术微调扩散模型。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合4D场景几何理解与扩散模型，实现几何一致且光照真实的视频对象插入。整体流程分为两个阶段：第一阶段是4D感知掩码生成，包括4D场景重建、用户控制对象放置和场景流驱动的对象传播；第二阶段是视频对象插入，先使用图像模型生成高质量的第一帧，然后扩散模型生成整个视频，同时处理光照和阴影等局部变化。此外，还构建了ROSE++数据集来支持训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 4D感知掩码生成模块，能处理复杂遮挡情况；2) 扩展视频修复模型，可合成对象周围的局部变化；3) ROSE++数据集，通过VLM生成参考图像；4) 场景流驱动的对象传播，确保物理一致性。相比之前工作，传统方法仅编辑给定掩码内区域，无法处理局部变化；其他方法依赖整个空间区域掩码，无法处理真实遮挡；GenProp等方法未考虑可见性随时间的变化，导致在遮挡场景下不一致。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'InsertAnywhere通过结合4D场景几何理解和扩散模型，实现了在复杂场景中几何一致且光照真实的高质量视频对象插入，显著优于现有商业生成工具。'}


### 论文摘要

Recent advances in diffusion-based video generation have opened new possibilities for controllable video editing, yet realistic video object insertion (VOI) remains challenging due to limited 4D scene understanding and inadequate handling of occlusion and lighting effects. We present InsertAnywhere, a new VOI framework that achieves geometrically consistent object placement and appearance-faithful video synthesis. Our method begins with a 4D aware mask generation module that reconstructs the scene geometry and propagates user specified object placement across frames while maintaining temporal coherence and occlusion consistency. Building upon this spatial foundation, we extend a diffusion based video generation model to jointly synthesize the inserted object and its surrounding local variations such as illumination and shading. To enable supervised training, we introduce ROSE++, an illumination aware synthetic dataset constructed by transforming the ROSE object removal dataset into triplets of object removed video, object present video, and a VLM generated reference image. Through extensive experiments, we demonstrate that our framework produces geometrically plausible and visually coherent object insertions across diverse real world scenarios, significantly outperforming existing research and commercial models.

---

## 45. GroundingME: Exposing the Visual Grounding Gap in MLLMs through Multi-Dimensional Evaluation

**论文链接:** [http://arxiv.org/abs/2512.17495v1](http://arxiv.org/abs/2512.17495v1)

**作者:** Rang Li, Lei Li, Shuhuai Ren, Hao Tian, Shuhao Gu, Shicheng Li, Zihao Yue, Yudong Wang, Wenhan Ma, Zhe Yang, Jingyuan Ma, Zhifang Sui, Fuli Luo

**发布时间:** 2025-12-19

### GPT解析

### 总结

该研究提出了GroundingME基准测试，用于评估多模态大语言模型(MLLMs)在视觉定位任务上的真实能力，发现当前模型与人类表现存在显著差距，并探索了两种改进策略。

### 背景

视觉定位是从自然语言描述中定位对象，是语言和视觉理解之间的关键桥梁。尽管MLLMs在现有基准测试上表现优异，但它们是否真正能以类人的复杂性将语言视觉化，还是仅在简化数据集上进行模式匹配仍存疑问。

### 目的

引入GroundingME基准测试，严格评估MLLMs的真实能力，系统性地挑战模型在四个关键维度上的表现，并探索改进策略。

### 方法

创建四个关键维度的基准测试：区分性(区分相似对象)、空间性(理解复杂关系)、有限性(处理遮挡或微小对象)和拒绝性(识别无法定位的查询)。通过自动生成和人工验证相结合，创建1005个具有挑战性的例子。评估25个最先进的MLLMs，并测试两种改进策略：测试时扩展和数据混合训练。

### 主要发现

最佳模型仅达到45.1%的准确率，大多数模型在拒绝任务上得分为0%，反射性地产生幻觉对象而非承认缺失，引发部署安全担忧。测试时扩展将复杂定位能力提高最多2.9%，数据混合训练将拒绝准确率从0%提高到27.9%。

### 结论

GroundingME既揭示了MLLMs的当前局限性，也是实现类人视觉定位的路线图。

### 翻译

视觉定位，即从自然语言描述中定位对象，代表了语言和视觉理解之间的关键桥梁。尽管多模态大语言模型在现有基准测试上取得了令人印象深刻的分数，但一个基本问题仍然存在：MLLMs是否真正能以类人的复杂性将语言视觉化，还是它们只是在简化的数据集上进行模式匹配？当前的基准测试无法捕捉真实世界的复杂性，而人类却能轻松处理模糊的引用并识别何时无法进行定位。为了严格评估MLLMs的真实能力，我们引入了GroundingME基准测试，该测试系统性地挑战模型在四个关键维度上的表现：(1)区分性，区分高度相似的对象；(2)空间性，理解复杂的关系描述；(3)有限性，处理遮挡或微小对象；(4)拒绝性，识别无法定位的查询。通过结合自动生成和人工验证的精心筛选，我们创建了1005个具有挑战性的例子，模拟真实世界的复杂性。评估25个最先进的MLLMs揭示了严重的能力差距：最佳模型仅达到45.1%的准确率，而大多数模型在拒绝任务上得分为0%，反射性地产生幻觉对象而不是承认它们的缺失，这引发了部署的关键安全问题。我们探索了两种改进策略：(1)测试时扩展通过思考轨迹选择最佳响应，将复杂定位能力提高最多2.9%；(2)数据混合训练教模型识别无法定位的查询，将拒绝准确率从0%提高到27.9%。因此，GroundingME既是一个诊断工具，揭示了MLLMs的当前局限性，也是实现类人视觉定位的路线图。


### 论文摘要

Visual grounding, localizing objects from natural language descriptions, represents a critical bridge between language and vision understanding. While multimodal large language models (MLLMs) achieve impressive scores on existing benchmarks, a fundamental question remains: can MLLMs truly ground language in vision with human-like sophistication, or are they merely pattern-matching on simplified datasets? Current benchmarks fail to capture real-world complexity where humans effortlessly navigate ambiguous references and recognize when grounding is impossible. To rigorously assess MLLMs' true capabilities, we introduce GroundingME, a benchmark that systematically challenges models across four critical dimensions: (1) Discriminative, distinguishing highly similar objects, (2) Spatial, understanding complex relational descriptions, (3) Limited, handling occlusions or tiny objects, and (4) Rejection, recognizing ungroundable queries. Through careful curation combining automated generation with human verification, we create 1,005 challenging examples mirroring real-world complexity. Evaluating 25 state-of-the-art MLLMs reveals a profound capability gap: the best model achieves only 45.1% accuracy, while most score 0% on rejection tasks, reflexively hallucinating objects rather than acknowledging their absence, raising critical safety concerns for deployment. We explore two strategies for improvements: (1) test-time scaling selects optimal response by thinking trajectory to improve complex grounding by up to 2.9%, and (2) data-mixture training teaches models to recognize ungroundable queries, boosting rejection accuracy from 0% to 27.9%. GroundingME thus serves as both a diagnostic tool revealing current limitations in MLLMs and a roadmap toward human-level visual grounding.

---

## 46. Video Detective: Seek Critical Clues Recurrently to Answer Question from Long Videos

**论文链接:** [http://arxiv.org/abs/2512.17229v1](http://arxiv.org/abs/2512.17229v1)

**作者:** Henghui Du, Chang Zhou, Chunjie Zhang, Xi Chen, Di Hu

**发布时间:** 2025-12-19

### GPT解析

### 总结

该研究提出了一种名为VideoDetective的高效问题感知记忆机制，使多模态大语言模型能够从长视频中反复寻求关键线索，有效处理长视频问答任务，同时减少内存消耗。

### 背景

长视频问答对多模态大语言模型构成重大挑战，因为长视频包含巨大的上下文和过载的信息，导致高昂的内存消耗。现有方法通过减少视觉标记或扩展模型上下文长度来解决这些问题，但可能会遗漏有用信息或需要大量计算。

### 目的

设计一种能够有效处理长视频问答任务的方法，使多模态大语言模型能够在有限内存条件下从大量信息中准确识别并利用关键线索。

### 方法

提出VideoDetective方法，通过迭代处理视频子片段实现任务简化。对每个子片段采用问题感知压缩策略，引入少量特殊记忆标记实现有目的的压缩，使模型在减少视觉标记的同时有效寻求关键线索。同时反复聚合和存储这些记忆标记以更新历史上下文，供后续子片段重用。此外，研究团队引入了GLVC数据集，用于更有效地衡量模型的长视频理解能力。

### 主要发现

实验结果表明，该方法使具有32K有限上下文长度的多模态大语言模型能够高效处理100K标记（3600帧，1fps采样的一小时视频），仅需2分钟和37GB GPU内存使用。多个长视频基准的评估结果表明，该方法能够从大量信息中更有效地寻求关键线索。

### 结论

VideoDetective通过问题感知记忆机制有效解决了长视频问答中的内存消耗和信息过载问题，使模型能够在有限资源条件下高效处理长视频内容，准确提取关键信息。

### 翻译

长视频问答由于巨大的上下文和过载的信息，为多模态大语言模型带来了重大挑战，这也可能导致高昂的内存消耗。虽然现有方法试图通过减少视觉标记或扩展模型上下文长度来解决这些问题，但它们可能会遗漏有用信息或需要大量计算。事实上，在回答给定问题时，只需要少量关键信息。因此，我们提出了一种高效的问题感知记忆机制，使多模态大语言模型能够反复寻求这些关键线索。我们的方法名为VideoDetective，通过迭代处理视频子片段来简化此任务。对于每个子片段，通过引入少量特殊记忆标记来实现问题感知压缩策略，从而实现有目的的压缩。这使得模型能够在减少视觉标记的同时有效寻求关键线索。然后，由于历史上下文可能有显著影响，我们反复聚合和存储这些记忆标记以更新历史上下文，该上下文将被后续子片段重用。此外，为了更有效地衡量模型的长视频理解能力，我们引入了GLVC，这是一个长视频问答数据集，其特点是定位散布在整个视频中的关键和具体线索。实验结果表明，我们的方法使具有32K有限上下文长度的多模态大语言模型能够高效处理100K标记（3600帧，1fps采样的一小时视频），仅需2分钟和37GB GPU内存使用。多个长视频基准的评估结果表明，我们的方法能够从大量信息中更有效地寻求关键线索。


### 论文摘要

Long Video Question-Answering (LVQA) presents a significant challenge for Multi-modal Large Language Models (MLLMs) due to immense context and overloaded information, which could also lead to prohibitive memory consumption. While existing methods attempt to address these issues by reducing visual tokens or extending model's context length, they may miss useful information or take considerable computation. In fact, when answering given questions, only a small amount of crucial information is required. Therefore, we propose an efficient question-aware memory mechanism, enabling MLLMs to recurrently seek these critical clues. Our approach, named VideoDetective, simplifies this task by iteratively processing video sub-segments. For each sub-segment, a question-aware compression strategy is employed by introducing a few special memory tokens to achieve purposefully compression. This allows models to effectively seek critical clues while reducing visual tokens. Then, due to history context could have a significant impact, we recurrently aggregate and store these memory tokens to update history context, which would be reused for subsequent sub-segments. Furthermore, to more effectively measure model's long video understanding ability, we introduce GLVC (Grounding Long Video Clues), a long video question-answering dataset, which features grounding critical and concrete clues scattered throughout entire videos. Experimental results demonstrate our method enables MLLMs with limited context length of 32K to efficiently process 100K tokens (3600 frames, an hour-long video sampled at 1fps), requiring only 2 minutes and 37GB GPU memory usage. Evaluation results across multiple long video benchmarks illustrate our method can more effectively seek critical clues from massive information.

---

## 47. A Benchmark and Agentic Framework for Omni-Modal Reasoning and Tool Use in Long Videos

**论文链接:** [http://arxiv.org/abs/2512.16978v1](http://arxiv.org/abs/2512.16978v1)

**作者:** Mohammed Irfan Kurpath, Jaseel Muhammad Kaithakkodan, Jinxing Zhou, Sahal Shaji Mullappilly, Mohammad Almansoori, Noor Ahsan, Beknur Kalmakhanbet, Sambal Shikhar, Rishabh Lalla, Jean Lahoud, Mariette Awad, Fahad Shahbaz Khan, Salman Khan, Rao Muhammad Anwer, Hisham Cholakkal

**发布时间:** 2025-12-18

### GPT解析

### 总结

该研究引入了LongSHOTBench，一个新的诊断基准测试，用于评估长形式多模态视频理解能力，同时提出了LongSHOTAgent智能体系统。研究显示当前最先进的MLLM在该基准测试上表现不佳，突显了长形式视频理解的挑战。

### 背景

长形式多模态视频理解需要整合视觉、语音和环境音频，并进行连贯的长程推理。现有基准测试要么强调时间长度，要么强调多模态丰富性，但很少同时兼顾两者，且主要依赖单一分数准确率，掩盖了失败模式。

### 目的

引入LongSHOTBench，一个包含开放式、意图驱动问题；单轮和多轮对话；以及需要跨视频、音频和语音进行多模态推理和智能体工具使用任务的诊断基准测试。

### 方法

LongSHOTBench通过可扩展、人工验证的流程生成，确保覆盖率和可重复性，所有样本都经过人工验证和纠正。同时提出LongSHOTAgent智能体系统，通过预处理、搜索和迭代分析来分析长视频。

### 主要发现

在LongSHOTBench上，最先进的MLLM表现差距明显：Gemini-2.5-Flash达到52.95%，开源模型低于30%，LongSHOTAgent达到44.66%。这些结果强调了现实世界长形式视频理解的难度。

### 结论

LongSHOTBench为评估和改进MLLMs提供了实用、可重复的基础，所有资源都在GitHub上提供。

### 翻译

长形式多模态视频理解需要整合视觉、语音和环境音频，并进行连贯的长程推理。现有基准测试要么强调时间长度，要么强调多模态丰富性，但很少同时兼顾两者。虽然一些基准测试包含开放式问题和高级指标，但它们主要依赖单一分数准确率，掩盖了失败模式。我们引入了LongSHOTBench，这是一个包含开放式、意图驱动问题的诊断基准；单轮和多轮对话；以及需要跨视频、音频和语音进行多模态推理和智能体工具使用的任务。每个项目都包含参考答案和分级评分标准，以便进行可解释和可追溯的评估。LongSHOTBench通过可扩展、人工验证的流程生成，以确保覆盖率和可重复性。我们LongSHOTBench中的所有样本都经过人工验证和纠正。此外，我们提出了LongSHOTAgent，一个通过预处理、搜索和迭代分析来分析长视频的智能体系统。在LongSHOTBench上，最先进的MLLM显示出很大差距：Gemini-2.5-Flash达到52.95%，开源模型低于30%，LongSHOTAgent达到44.66%。这些结果强调了现实世界长形式视频理解的难度。LongSHOTBench为评估和改进MLLMs提供了实用、可重复的基础。所有资源都在GitHub上提供：https://github.com/mbzuai-oryx/longshot。


### 论文摘要

Long-form multimodal video understanding requires integrating vision, speech, and ambient audio with coherent long-range reasoning. Existing benchmarks emphasize either temporal length or multimodal richness, but rarely both and while some incorporate open-ended questions and advanced metrics, they mostly rely on single-score accuracy, obscuring failure modes. We introduce LongShOTBench, a diagnostic benchmark with open-ended, intent-driven questions; single- and multi-turn dialogues; and tasks requiring multimodal reasoning and agentic tool use across video, audio, and speech. Each item includes a reference answer and graded rubric for interpretable, and traceable evaluation. LongShOTBench is produced via a scalable, human-validated pipeline to ensure coverage and reproducibility. All samples in our LongShOTBench are human-verified and corrected. Furthermore, we present LongShOTAgent, an agentic system that analyzes long videos via preprocessing, search, and iterative refinement. On LongShOTBench, state-of-the-art MLLMs show large gaps: Gemini-2.5-Flash achieves 52.95%, open-source models remain below 30%, and LongShOTAgent attains 44.66%. These results underscore the difficulty of real-world long-form video understanding. LongShOTBench provides a practical, reproducible foundation for evaluating and improving MLLMs. All resources are available on GitHub: https://github.com/mbzuai-oryx/longshot.

---

## 48. LiteGE: Lightweight Geodesic Embedding for Efficient Geodesics Computation and Non-Isometric Shape Correspondence

**论文链接:** [http://arxiv.org/abs/2512.17781v1](http://arxiv.org/abs/2512.17781v1)

**作者:** Yohanes Yudhi Adikusuma, Qixing Huang, Ying He

**发布时间:** 2025-12-19

### GPT解析

### 总结

本文提出了一种名为LiteGE的轻量级方法，用于计算3D表面上的测地距离，解决了现有方法内存占用高、延迟大的问题。

### 背景

计算3D表面上的测地距离是3D视觉和几何处理的基础任务，与形状对应等密切相关。现有基于学习的方法表现良好但依赖大型3D骨干网络，导致高内存使用和高延迟，限制了在交互式或资源受限环境中的应用。

### 目的

开发一种轻量级方法，减少内存使用和推理时间，使其能够适用于交互式或资源受限的应用场景。

### 方法

LiteGE通过对有意义的体素处的无符号距离场(UDF)样本应用主成分分析(PCA)，构建紧凑的、类别感知的形状描述符。这种方法计算效率高，不需要高容量网络，且在稀疏点云上保持鲁棒性，支持仅300个点的输入。

### 主要发现

与现有神经方法相比，LiteGE将内存使用和推理时间减少了高达300倍；通过利用测地距离和形状对应的内在关系，实现了快速准确的形状匹配；与最先进的基于网格的方法相比，实现了高达1000倍的加速，同时在非等距形状对上保持相当的准确性。

### 结论

LiteGE是一种高效的方法，能够在保持准确性的同时大幅减少计算资源需求，特别适合资源受限的应用场景，包括点云输入。

### 翻译

计算3D表面上的测地距离是3D视觉和几何处理中许多任务的基础，与形状对应等任务有密切关系。最近的基于学习的方法取得了强大性能，但依赖于大型3D骨干网络，导致高内存使用和延迟，限制了它们在交互式或资源受限环境中的使用。我们介绍了LiteGE，一种轻量级方法，通过对有意义的体素处的无符号距离场(UDF)样本应用PCA，构建紧凑的、类别感知的形状描述符。这种描述符计算效率高，不需要高容量网络。LiteGE在稀疏点云上保持鲁棒性，支持仅300个点的输入，而先前方法在此情况下失败。大量实验表明，与现有神经方法相比，LiteGE将内存使用和推理时间减少了高达300倍。此外，通过利用测地距离和形状对应之间的内在关系，LiteGE实现了快速准确的形状匹配。与最先进的基于网格的方法相比，我们的方法实现了高达1000倍的加速，同时在非等距形状对上保持相当的准确性，包括在点云输入上的评估。


### 论文摘要

Computing geodesic distances on 3D surfaces is fundamental to many tasks in 3D vision and geometry processing, with deep connections to tasks such as shape correspondence. Recent learning-based methods achieve strong performance but rely on large 3D backbones, leading to high memory usage and latency, which limit their use in interactive or resource-constrained settings. We introduce LiteGE, a lightweight approach that constructs compact, category-aware shape descriptors by applying PCA to unsigned distance field (UDFs) samples at informative voxels. This descriptor is efficient to compute and removes the need for high-capacity networks. LiteGE remains robust on sparse point clouds, supporting inputs with as few as 300 points, where prior methods fail. Extensive experiments show that LiteGE reduces memory usage and inference time by up to 300$\times$ compared to existing neural approaches. In addition, by exploiting the intrinsic relationship between geodesic distance and shape correspondence, LiteGE enables fast and accurate shape matching. Our method achieves up to 1000$\times$ speedup over state-of-the-art mesh-based approaches while maintaining comparable accuracy on non-isometric shape pairs, including evaluations on point-cloud inputs.

---

## 49. UniStateDLO: Unified Generative State Estimation and Tracking of Deformable Linear Objects Under Occlusion for Constrained Manipulation

**论文链接:** [http://arxiv.org/abs/2512.17764v1](http://arxiv.org/abs/2512.17764v1)

**作者:** Kangchen Lv, Mingrui Yu, Shihefeng Wang, Xiangyang Ji, Xiang Li

**发布时间:** 2025-12-19

**备注:** The first two authors contributed equally. Project page: https://unistatedlo.github.io

### GPT解析

### 总结

本文提出了UniStateDLO，这是第一个完整的可变形线性物体(DLOs)感知管道，使用深度学习方法在严重遮挡情况下实现鲁棒性能。该方法将单帧状态估计和跨帧状态跟踪都表述为条件生成问题，利用扩散模型处理部分观测和高维DLO状态之间的复杂映射。UniStateDLO能够处理各种遮挡模式，仅使用合成数据训练即可实现零样本仿真到现实的泛化，并在实验中优于所有最先进方法。

### 背景

基于视觉的可变形线性物体感知方法虽然已被广泛探索，但在受约束操作环境中容易受到遮挡影响，这些环境通常由周围障碍物、大型和变化的变形以及有限视角引起。此外，状态空间的高维性、缺乏明显的视觉特征以及传感器噪声的存在进一步增加了可靠DLO感知的挑战。

### 目的

解决现有DLO感知方法在遮挡环境下的脆弱性问题，包括周围障碍物导致的遮挡、大型和变化的变形、有限的视角、高维状态空间、缺乏明显视觉特征以及传感器噪声等问题。

### 方法

提出UniStateDLO，将单帧状态估计和跨帧状态跟踪都表述为条件生成问题，利用扩散模型捕获部分观测和高维DLO状态之间的复杂映射能力。仅使用大规模合成数据训练整个网络，实现零样本仿真到现实的泛化，无需任何真实世界训练数据。

### 主要发现

1) UniStateDLO能有效处理各种遮挡模式，包括初始遮挡、自遮挡和由多个物体引起的遮挡；2) 该方法表现出强大的数据效率，仅通过合成数据训练即可实现零样本仿真到现实的泛化；3) 在模拟和真实世界实验中，UniStateDLO在估计和跟踪方面都优于所有最先进的基线方法；4) 即使在大量遮挡的情况下，也能实时生成全局平滑且局部精确的DLO状态预测；5) 作为闭环DLO操作系统的前端模块集成时，能在复杂受限的3D环境中支持稳定的反馈控制。

### 结论

UniStateDLO通过创新的深度学习方法解决了DLO感知在遮挡环境中的关键挑战，实现了高性能的DLO状态估计和跟踪。该方法不仅在各种遮挡条件下表现出色，还通过仅使用合成数据进行训练实现了零样本仿真到现实的泛化，大大降低了数据收集的成本和复杂性。其作为前端模块在闭环DLO操作系统中的成功集成进一步验证了其在实际应用中的价值。

### 翻译

可变形线性物体(DLOs)（如电缆、绳索和电线）的感知是成功下游操作的基础。尽管基于视觉的方法已被广泛探索，但它们在受约束操作环境中由于周围障碍物、大型和变化的变形以及有限视角而常见遮挡的情况下仍然非常脆弱。此外，状态空间的高维性、缺乏明显的视觉特征以及传感器噪声的存在进一步增加了可靠DLO感知的挑战。为解决这些开放性问题，本文提出了UniStateDLO，这是第一个使用深度学习方法的完整DLO感知管道，能够在严重遮挡下实现鲁棒性能，涵盖了从部分点云的单帧状态估计和跨帧状态跟踪。这两个任务都被表述为条件生成问题，利用扩散模型捕获部分观测和高维DLO状态之间复杂映射的强大能力。UniStateDLO能够有效处理各种遮挡模式，包括初始遮挡、自遮挡和由多个物体引起的遮挡。此外，由于整个网络仅在大规模合成数据集上训练，它表现出强大的数据效率，能够在没有任何真实世界训练数据的情况下实现零样本仿真到现实的泛化。全面的模拟和真实世界实验表明，UniStateDLO在估计和跟踪方面都优于所有最先进的基线方法，即使在大量遮挡的情况下也能实时产生全局平滑且局部精确的DLO状态预测。作为闭环DLO操作系统中的前端模块集成，它进一步展示了在复杂受限3D环境中支持稳定反馈控制的能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决可变形线性物体（如电缆、绳索、电线）在遮挡情况下的三维状态估计和跟踪问题。这个问题在现实中非常重要，因为准确感知DLO的形状和位置是机器人成功操作它们的基础，在制造、服务和医疗等领域有广泛应用。然而，在受限操作环境中，常见的遮挡问题（由障碍物、物体自遮挡或复杂变形引起）使得现有视觉方法难以可靠地估计DLO状态，这限制了机器人在复杂环境中的操作能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有DLO感知方法的局限性：单帧估计方法忽略时间连续性，而跟踪方法严重依赖准确初始化。受扩散模型在处理复杂概率分布方面能力的启发，作者将DLO状态估计和跟踪都表述为条件生成任务。他们设计了一个两分支网络架构，一个利用全局信息实现遮挡鲁棒性，另一个利用局部信息确保节点级别的精确估计，最后通过扩散模型融合这些预测。该方法借鉴了扩散模型在图像生成、人体姿态估计等领域的应用，以及PointNet++用于点云特征提取和非刚性点集配准算法的思路。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将DLO状态估计和跟踪都表述为条件生成问题，利用扩散模型捕捉部分观测和高维DLO状态之间复杂的映射关系。整体流程包括：1) 单帧状态估计模块：输入部分点云，通过点云标准化、PointNet++特征提取、两分支处理（直接回归和点到点投票）以及扩散模型融合生成最终三维节点预测；2) 跨帧状态跟踪模块：利用当前点云和前一帧节点估计，通过KNN特征聚合和扩散模型预测节点运动；3) 预处理和后处理：包括点云标准化、B样条拟合和跟踪失败检测机制。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 统一的DLO感知管道，首次同时支持单帧估计和跨帧跟踪；2) 条件生成方法，利用扩散模型解决遮挡情况下的节点级别不确定性；3) 零样本模拟到现实泛化能力，仅在合成数据上训练即可泛化到真实世界。相比之前工作，该方法不依赖手动设计的配准参数，不需要多视图相机或仿真引擎，在严重遮挡下表现更鲁棒，能处理更复杂的变形和拓扑结构，且计算效率更高。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'UniStateDLO通过统一的条件生成框架，利用扩散模型解决了可变形线性物体在严重遮挡情况下的三维状态估计和跟踪问题，实现了高精度、鲁棒性和实时性能，并在零样本模拟到现实泛化中表现出色。'}


### 论文摘要

Perception of deformable linear objects (DLOs), such as cables, ropes, and wires, is the cornerstone for successful downstream manipulation. Although vision-based methods have been extensively explored, they remain highly vulnerable to occlusions that commonly arise in constrained manipulation environments due to surrounding obstacles, large and varying deformations, and limited viewpoints. Moreover, the high dimensionality of the state space, the lack of distinctive visual features, and the presence of sensor noises further compound the challenges of reliable DLO perception. To address these open issues, this paper presents UniStateDLO, the first complete DLO perception pipeline with deep-learning methods that achieves robust performance under severe occlusion, covering both single-frame state estimation and cross-frame state tracking from partial point clouds. Both tasks are formulated as conditional generative problems, leveraging the strong capability of diffusion models to capture the complex mapping between highly partial observations and high-dimensional DLO states. UniStateDLO effectively handles a wide range of occlusion patterns, including initial occlusion, self-occlusion, and occlusion caused by multiple objects. In addition, it exhibits strong data efficiency as the entire network is trained solely on a large-scale synthetic dataset, enabling zero-shot sim-to-real generalization without any real-world training data. Comprehensive simulation and real-world experiments demonstrate that UniStateDLO outperforms all state-of-the-art baselines in both estimation and tracking, producing globally smooth yet locally precise DLO state predictions in real time, even under substantial occlusions. Its integration as the front-end module in a closed-loop DLO manipulation system further demonstrates its ability to support stable feedback control in complex, constrained 3-D environments.

---

## 50. 论文ID: 2512.17568v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.17568v1.json'

---

## 51. Voxel-GS: Quantized Scaffold Gaussian Splatting Compression with Run-Length Coding

**论文链接:** [http://arxiv.org/abs/2512.17528v1](http://arxiv.org/abs/2512.17528v1)

**作者:** Chunyang Fu, Xiangrui Liu, Shiqi Wang, Zhu Li

**发布时间:** 2025-12-19

**备注:** Accepted by DCC 2026

### GPT解析

### 总结

本文提出了Voxel-GS，一个简单而高效的框架，用于压缩高斯散射格式点云，使用轻量级速率代理和游程编码实现竞争性性能。

### 背景

高斯散射格式点云需要有效的压缩处理。

### 目的

提出一个简单但高效的框架Voxel-GS，用于压缩高斯散射点云。

### 方法

使用可微分量化将Scaffold-GS的高斯属性离散化，设计基于拉普拉斯的速率代理施加熵约束，最后使用八叉树和游程编码对整数型高斯点云进行无损压缩。

### 主要发现

所提出的速率代理能够准确估计游程编码的比特率，使Voxel-GS能够消除冗余并优化更紧凑的表示。

### 结论

该方法实现了显著的压缩比，并且比先前技术具有更快的编码速度。

### 翻译

大规模高斯散射格式点云需要有效的压缩。在本文中，我们提出了Voxel-GS，一个简单而高效的框架，它脱离了先前工作中复杂的神经熵模型，而是仅使用轻量级速率代理和游程编码实现竞争性性能。具体而言，我们采用可微分量化将Scaffold-GS的高斯属性离散化。随后，设计了一种基于拉普拉斯的速率代理来施加熵约束，指导生成高保真和紧凑的重建。最后，使用八叉树和游程编码对这种整数型高斯点云进行无损压缩。实验验证了所提出的速率代理能够准确估计游程编码的比特率，使Voxel-GS能够消除冗余并优化更紧凑的表示。因此，我们的方法实现了显著的压缩比，并且比先前技术具有更快的编码速度。代码可在https://github.com/zb12138/VoxelGS获取。


### 论文摘要

Substantial Gaussian splatting format point clouds require effective compression. In this paper, we propose Voxel-GS, a simple yet highly effective framework that departs from the complex neural entropy models of prior work, instead achieving competitive performance using only a lightweight rate proxy and run-length coding. Specifically, we employ a differentiable quantization to discretize the Gaussian attributes of Scaffold-GS. Subsequently, a Laplacian-based rate proxy is devised to impose an entropy constraint, guiding the generation of high-fidelity and compact reconstructions. Finally, this integer-type Gaussian point cloud is compressed losslessly using Octree and run-length coding. Experiments validate that the proposed rate proxy accurately estimates the bitrate of run-length coding, enabling Voxel-GS to eliminate redundancy and optimize for a more compact representation. Consequently, our method achieves a remarkable compression ratio with significantly faster coding speeds than prior art. The code is available at https://github.com/zb12138/VoxelGS.

---

## 52. Delaunay-Rips filtration: a study and an algorithm

**论文链接:** [http://arxiv.org/abs/2512.17382v1](http://arxiv.org/abs/2512.17382v1)

**作者:** Mattéo Clémot, Julie Digne, Julien Tierny

**发布时间:** 2025-12-19

### GPT解析

### 总结

这篇论文介绍了Delaunay-Rips过滤，一种在低维欧几里得点云中比传统Rips过滤更轻量、更快速的替代方法，并提供了全面的理论和经验分析。

### 背景

Delaunay-Rips过滤是一种比众所周知的Rips过滤更轻量、更快速的替代方法，特别适用于低维欧几里得点云。然而，尽管有这些优势，它很少被研究。

### 目的

论文旨在弥合这一研究空白，为Delaunay-Rips过滤提供全面的理论和经验分析。

### 方法

从理论角度分析Delaunay-Rips持久图对Rips持久图的近似性，描述当输入点云被扰动时Delaunay-Rips持久图的不稳定性，并引入一个在任意维度计算Delaunay-Rips过滤持久图的算法。

### 主要发现

Delaunay-Rips持久图可以近似Rips持久图；当输入点云被扰动时，Delaunay-Rips持久图存在不稳定性；作者提出的新算法在低维中比传统方法更快且内存占用更少。

### 结论

Delaunay-Rips过滤是低维欧几里得点云中Rips过滤的有效替代方案，具有计算和内存效率优势。作者提供了C++实现和Python绑定可供使用。

### 翻译

Delaunay-Rips过滤是众所周知的Rips过滤在低维欧几里得点云中的一种更轻量、更快速的替代方法。尽管有这些优势，它很少被研究。在本文中，我们旨在通过提供对这种构造的全面理论和经验分析来弥合这一差距。从理论角度来看，我们展示了与Delaunay-Rips过滤相关的持久图如何近似于使用Rips过滤获得的持久图。此外，我们描述了当输入点云被扰动时，Delaunay-Rips持久图的不稳定性。最后，我们引入了一个在任意维度计算Delaunay-Rips过滤持久图的算法。我们证明，我们的方法在低维中比传统方法更快且内存占用更少。我们的C++实现（带有Python绑定）可在https://github.com/MClemot/GeoPH获取。


### 论文摘要

The Delaunay-Rips filtration is a lighter and faster alternative to the well-known Rips filtration for low-dimensional Euclidean point clouds. Despite these advantages, it has seldom been studied. In this paper, we aim to bridge this gap by providing a thorough theoretical and empirical analysis of this construction. From a theoretical perspective, we show how the persistence diagrams associated with the Delaunay-Rips filtration approximate those obtained with the Rips filtration. Additionally, we describe the instabilities of the Delaunay-Rips persistence diagrams when the input point cloud is perturbed. Finally, we introduce an algorithm that computes persistence diagrams of Delaunay-Rips filtrations in any dimension. We show that our method is faster and has a lower memory footprint than traditional approaches in low dimensions. Our C++ implementation, which comes with Python bindings, is available at https://github.com/MClemot/GeoPH.

---

## 53. Fully Dynamic Algorithms for Chamfer Distance

**论文链接:** [http://arxiv.org/abs/2512.16639v2](http://arxiv.org/abs/2512.16639v2)

**作者:** Gramoz Goranci, Shaofeng Jiang, Peter Kiss, Eva Szilagyi, Qiaoyuan Yang

**发布时间:** 2025-12-18

**备注:** NeurIPS 2025

### GPT解析

### 总结

研究在完全动态设置下计算Chamfer距离的问题，提出首个在ℓ_p范数下维护Chamfer距离近似的动态算法。

### 背景

Chamfer距离是点云广泛使用的差异性度量，在机器学习等领域有实际应用，常需要在对动态变化的数据集进行重复评估时使用。

### 目的

高效维护对动态变化点集A和B的Chamfer距离的近似值，其中点集通过插入或删除操作动态变化。

### 方法

提出一种近似最近邻搜索的动态算法，该算法在更新Chamfer距离近似值时只需较少的开销。

### 主要发现

算法可在tilde(O)(ε^{-d})更新时间内获得(1+ε)-近似，或在tilde(O)(d n^{ε^2} ε^{-4})更新时间内获得O(1/ε)-近似。

### 结论

在真实数据集上的评估表明，该方法与自然基线方法相比具有竞争力。

### 翻译

我们研究在完全动态设置下计算Chamfer距离的问题，其中两个点集A、B⊂ℝ^d（每个大小最多为n）通过点的插入或删除动态变化，目标是高效维护对dist_CH(A,B)=∑_{a∈A} min_{b∈B} dist(a,b)的近似，其中dist是一种距离度量。Chamfer距离是点云广泛使用的差异性度量，在许多需要重复评估动态变化数据集的实际应用中都有应用，例如在机器学习中用作损失函数。在本文中，我们首次提出了在ℓ_p范数下（p∈{1,2}）维护Chamfer距离近似的动态算法。我们的算法近似于最近邻搜索，只需少量开销。代入标准的ANN界限，我们在tilde(O)(ε^{-d})更新时间内获得(1+ε)-近似，在tilde(O)(d n^{ε^2} ε^{-4})更新时间内获得O(1/ε)-近似。我们在真实数据集上评估了我们的方法，并证明其与自然基线方法相比具有竞争力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在动态变化的点集上高效计算Chamfer距离的问题。当两个点集通过插入或删除点不断变化时，如何高效维护它们之间的Chamfer距离近似值。这个问题在现实中非常重要，因为Chamfer距离是点云间常用的不相似性度量，广泛应用于机器学习（如作为损失函数）、计算机视觉（如3D物体重建）和医学成像（如跟踪解剖结构变化）等领域。在这些应用中，点云数据经常动态变化，需要持续评估点云间的相似性，而现有静态算法在每次更新后重新计算的效率太低。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先认识到Chamfer距离计算本质上可以转化为近似最近邻搜索问题，因为每个点到另一个点集的距离就是该点到点集中最近点的距离。他们借鉴了现有的动态最近邻数据结构作为子程序，并改进了静态Chamfer距离算法中的重要性采样框架。在动态场景中，作者发现无法显式维护点之间的近似分配关系，因为这会导致每次更新时大量点的分配关系发生变化。因此，他们设计了一种隐式表示距离估计的方法，通过动态四叉树结构来维护每个点的'匹配单元格'信息，从而高效地估计距离而不显式存储所有分配关系。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是利用重要性采样和空间划分来高效估计Chamfer距离。具体来说，算法使用动态四叉树将空间递归划分为嵌套单元格，对于每个点a∈A，确定其'匹配单元格'（同时包含a和B中某个点的最小单元格），并利用单元格大小作为距离的近似估计。整体流程分为三部分：1) 数据结构维护：在四叉树节点中存储A点数、B点数和匹配点数；2) 更新处理：当点插入或删除时，更新受影响路径上的节点信息和采样器；3) 查询处理：通过采样单元格而非直接采样点，对采样出的点使用近似最近邻查询获取距离估计，最后通过重要性采样公式组合多个样本结果得到Chamfer距离的估计。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首个动态Chamfer距离算法，能够在点集变化时高效维护距离近似值；2) 将Chamfer距离计算问题转化为近似最近邻搜索问题；3) 设计隐式距离估计方法，通过四叉树结构中的匹配信息实现；4) 提供多种参数配置下的高效更新方案。相比之前的工作，不同之处在于：静态算法每次更新需重新计算，复杂度高达O(nd·ε^{-2})；动态EMD算法仅适用于二维空间且只能达到O(1/ε)近似；本文方法适用于任意维度，在低维可实现(1+ε)近似，且证明了维护点集映射关系的下界，表明无法高效维护精确分配关系。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文首次提出了在动态变化的点集上高效计算Chamfer距离的算法，通过将问题转化为近似最近邻搜索并利用重要性采样，实现了比重新计算高得多的效率，为处理动态点云的应用提供了实用工具。'}


### 论文摘要

We study the problem of computing Chamfer distance in the fully dynamic setting, where two set of points $A, B \subset \mathbb{R}^{d}$, each of size up to $n$, dynamically evolve through point insertions or deletions and the goal is to efficiently maintain an approximation to $\mathrm{dist}_{\mathrm{CH}}(A,B) = \sum_{a \in A} \min_{b \in B} \textrm{dist}(a,b)$, where $\textrm{dist}$ is a distance measure. Chamfer distance is a widely used dissimilarity metric for point clouds, with many practical applications that require repeated evaluation on dynamically changing datasets, e.g., when used as a loss function in machine learning. In this paper, we present the first dynamic algorithm for maintaining an approximation of the Chamfer distance under the $\ell_p$ norm for $p \in \{1,2 \}$. Our algorithm reduces to approximate nearest neighbor (ANN) search with little overhead. Plugging in standard ANN bounds, we obtain $(1+ε)$-approximation in $\tilde{O}(ε^{-d})$ update time and $O(1/ε)$-approximation in $\tilde{O}(d n^{ε^2} ε^{-4})$ update time. We evaluate our method on real-world datasets and demonstrate that it performs competitively against natural baselines.

---

## 54. 论文ID: 2512.16950v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.16950v1.json'

---

