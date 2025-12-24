# 今日论文推荐 - 2025-12-24

共 45 篇论文

---

## 1. LiteFusion: Taming 3D Object Detectors from Vision-Based to Multi-Modal with Minimal Adaptation

**论文链接:** [http://arxiv.org/abs/2512.20217v1](http://arxiv.org/abs/2512.20217v1)

**作者:** Xiangxuan Ren, Zhongdao Wang, Pin Tang, Guoqing Wang, Jilai Zheng, Chao Ma

**发布时间:** 2025-12-23

**备注:** 13 pages, 9 figures, 8 tables

### GPT解析

### 总结

本文提出了一种名为LiteFusion的新型多模态3D检测器，通过重新考虑LiDAR在相机-LiDAR融合范式中的作用，解决了现有方法对LiDAR的过度依赖问题，实现了在多样化硬件平台上的友好部署。

### 背景

3D物体检测对安全稳健的智能交通系统至关重要。当前多模态3D物体检测器通常依赖复杂架构和训练策略来获得更高检测精度，但这些方法严重依赖LiDAR传感器，导致在LiDAR缺失时性能大幅下降，同时难以在NPU和FPGA等多样化硬件平台上部署。

### 目的

解决多模态3D物体检测器对LiDAR的过度依赖问题，提高检测系统在实际场景中的鲁棒性和安全性，创建一个能在多样化硬件平台上部署的检测器。

### 方法

重新考虑LiDAR在相机-LiDAR融合范式中的作用，提出LiteFusion检测器，将LiDAR点云视为增强基于相机检测的几何信息的补充来源，而不是独立模态。在四元数空间内将LiDAR的补充特征集成到图像特征中，利用正交约束保留模态间的关系，形成紧凑的跨模态嵌入。

### 主要发现

在nuScenes数据集上，LiteFusion将基线基于视觉的检测器的mAP提高了20.4%，NDS提高了19.7%，参数仅增加1.1%，且无需专用的LiDAR编码器。即使在没有LiDAR输入的情况下，LiteFusion也能保持良好的结果。

### 结论

LiteFusion具有有利的鲁棒性和有效性，适用于多样化的融合范式和部署场景，通过消除对3D骨干网络的依赖，提高了部署友好性。

### 翻译

3D物体检测对安全稳健的智能交通系统至关重要。当前多模态3D物体检测器通常依赖复杂架构和训练策略来获得更高检测精度。然而，这些方法严重依赖LiDAR传感器，导致在LiDAR缺失时性能大幅下降，这损害了自主系统在实际场景中的鲁棒性和安全性。此外，由于依赖主要针对NVIDIA GPU优化的3D稀疏卷积算子，现有多模态检测器难以在NPU和FPGA等多样化硬件平台上部署。为解决这些挑战，我们重新考虑了LiDAR在相机-LiDAR融合范式中的作用，并引入了一种新型的多模态3D检测器LiteFusion。LiteFusion不将LiDAR点云视为具有独立特征提取骨干的独立模态，而是利用LiDAR数据作为增强基于相机检测的几何信息的补充来源。这种直接的方法完全消除了对3D骨干网络的依赖，使该方法具有高度部署友好性。具体而言，LiteFusion在四元数空间内将LiDAR的补充特征集成到图像特征中，在网络训练期间保持良好的正交约束。这有助于建模跨模态的特定领域关系，产生紧凑的跨模态嵌入。在nuScenes数据集上的实验表明，LiteFusion将基线基于视觉的检测器的mAP提高了20.4%，NDS提高了19.7%，参数仅增加1.1%，且无需专用的LiDAR编码器。值得注意的是，即使在没有LiDAR输入的情况下，LiteFusion也能保持良好的结果，突显了其在多样化融合范式和部署场景中有利的鲁棒性和有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有多模态3D目标检测器过度依赖LiDAR传感器的问题，以及难以部署在多样化硬件平台上的挑战。当LiDAR数据缺失时，现有方法性能大幅下降，影响自动驾驶系统的鲁棒性和安全性；同时，它们依赖专为NVIDIA GPU优化的3D稀疏卷积算子，在NPU、FPGA等其他平台上部署困难。这些问题限制了自动驾驶系统在实际生产环境中的应用和可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者重新思考了LiDAR在相机-LiDAR融合范式中的角色，提出将LiDAR数据视为增强相机检测的补充几何信息来源，而非独立模态。他们从相机检测器出发，用最小结构调整使其适应多模态检测，并开发了策略解决3D LiDAR几何与2D视觉信息间的域差距。方法借鉴了四元数代数处理跨模态融合，使用BEVFormer作为基础相机检测器，并参考了现有的特征融合方法，但提出了不同的单流架构。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将LiDAR数据视为增强相机检测的补充几何信息来源，而非需要单独特征提取的独立模态，消除对3D特征提取骨干网络的依赖。整体流程包括：1)使用渐进式响应框架将双流网络统一为单流；2)设计LiDAR几何集成器(LGI)，包含深度感知嵌入(DAE)和几何感知嵌入(GAE)；3)在DAE中使用四元数特征适应(Qua-FA)建模正交关系；4)将LiDAR数据投影到PV和BEV格式，通过模块生成几何特征并层层整合到相机特征中；5)增强的特征传递到检测头生成结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的相机辅助LiDAR融合方案，消除复杂的点云骨干网络；2)四元数空间嵌入实现参数高效的跨模态融合；3)渐进式响应框架逐步集成几何信息；4)完全基于标准算子，无需3D稀疏卷积。相比之前工作，LiteFusion使用单流而非双流架构，重新定义LiDAR角色，无需3D稀疏卷积，在LiDAR缺失时保持更强鲁棒性，且参数效率更高(仅增加1.1%参数)。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LiteFusion通过重新定义LiDAR角色和使用四元数空间融合，实现了无需3D稀疏卷积的高效、鲁棒且易于部署的相机-LiDAR融合框架，在保持最小参数增加的同时显著提升了3D目标检测性能。'}


### 论文摘要

3D object detection is fundamental for safe and robust intelligent transportation systems. Current multi-modal 3D object detectors often rely on complex architectures and training strategies to achieve higher detection accuracy. However, these methods heavily rely on the LiDAR sensor so that they suffer from large performance drops when LiDAR is absent, which compromises the robustness and safety of autonomous systems in practical scenarios. Moreover, existing multi-modal detectors face difficulties in deployment on diverse hardware platforms, such as NPUs and FPGAs, due to their reliance on 3D sparse convolution operators, which are primarily optimized for NVIDIA GPUs. To address these challenges, we reconsider the role of LiDAR in the camera-LiDAR fusion paradigm and introduce a novel multi-modal 3D detector, LiteFusion. Instead of treating LiDAR point clouds as an independent modality with a separate feature extraction backbone, LiteFusion utilizes LiDAR data as a complementary source of geometric information to enhance camera-based detection. This straightforward approach completely eliminates the reliance on a 3D backbone, making the method highly deployment-friendly. Specifically, LiteFusion integrates complementary features from LiDAR points into image features within a quaternion space, where the orthogonal constraints are well-preserved during network training. This helps model domain-specific relations across modalities, yielding a compact cross-modal embedding. Experiments on the nuScenes dataset show that LiteFusion improves the baseline vision-based detector by +20.4% mAP and +19.7% NDS with a minimal increase in parameters (1.1%) without using dedicated LiDAR encoders. Notably, even in the absence of LiDAR input, LiteFusion maintains strong results , highlighting its favorable robustness and effectiveness across diverse fusion paradigms and deployment scenarios.

---

## 2. The Seismic Wavefield Common Task Framework

**论文链接:** [http://arxiv.org/abs/2512.19927v1](http://arxiv.org/abs/2512.19927v1)

**作者:** Alexey Yermakov, Yue Zhao, Marine Denolle, Yiyu Ni, Philippe M. Wyder, Judah Goldfeder, Stefano Riva, Jan Williams, David Zoro, Amy Sara Rude, Matteo Tomasetto, Joe Germany, Joseph Bakarji, Georg Maierhofer, Miles Cranmer, J. Nathan Kutz

**发布时间:** 2025-12-22

**备注:** 35 pages, 7 figures

### GPT解析

### 总结

该研究提出了一个地震波场机器学习的通用任务框架(CTF)，旨在解决地震学在状态预测和重建方面的挑战，以及管理参数变异性问题。该框架提供结构化和严格的评估基础，包含多种规模的数据集和特定任务指标，有助于提高科学机器学习的严格性和可重复性。

### 背景

地震学在状态预测和重建（如地震早期预警和地面运动预测）以及管理源位置、机制和地球模型的参数变异性方面面临基本挑战。模拟方法受到大规模数据量和复杂性的限制，而实际数据工作则受到不充分反映地球复杂性的模型和稀疏传感器测量的限制。

### 目的

引入一个通用任务框架(CTF)用于地震波场的机器学习，提供结构化和严格的算法评估基础，取代临时的比较方法，提高科学机器学习的严格性和可重复性。

### 方法

开发了一个包含三个不同波场数据集的CTF，提供全球、地壳和局部等多种规模的精选数据集，以及涵盖预测、重建和现实约束下泛化的特定任务指标。受自然语言处理等领域CTF的启发，该框架为算法的面对面评估提供结构化基础。研究展示了两个数据集的评估程序，报告了各种方法和基础模型在重建地震波场方面的性能。

### 主要发现

CTF分数揭示了不同方法的优点、局限性和特定问题类的适用性，为地震波场机器学习提供了有价值的基准评估。

### 结论

通过在隐藏测试集上进行标准化评估，CTF框架有助于提高地震学机器学习研究的严格性和可重复性，取代了以往临时的比较方法。

### 翻译

地震学在状态预测和重建（例如地震早期预警和地面运动预测）以及管理源位置、机制和地球模型（例如地下结构和地形效应）的参数变异性方面面临基本挑战。通过模拟解决这些问题受到合成数据量和数值复杂性的大规模限制，而实际数据工作则受到无法充分反映地球复杂性的模型和现场稀疏传感器测量的限制。最近的机器学习(ML)努力有希望，但进展因缺乏适当的表征、公平的报告和严格的比较而模糊不清。为此，我们引入了一个用于地震波场机器学习的通用任务框架(CTF)，从三个不同的波场数据集开始。我们的CTF包含各种规模（全球、地壳和局部）的精选数据集集和特定任务的指标，涵盖预测、重建和现实约束（如噪声和有限数据）下的泛化。受自然语言处理等领域CTF的启发，该框架为算法的面对面评估提供结构化和严格的基础。我们通过报告两个数据集的分数来说明评估程序，展示了各种方法和基础模型在从模拟和真实世界传感器测量重建地震波场的性能。CTF分数揭示了不同方法的优点、局限性和特定问题类的适用性。我们的愿景是用隐藏测试集上的标准化评估取代临时的比较，提高科学机器学习的严格性和可重复性。


### 论文摘要

Seismology faces fundamental challenges in state forecasting and reconstruction (e.g., earthquake early warning and ground motion prediction) and managing the parametric variability of source locations, mechanisms, and Earth models (e.g., subsurface structure and topography effects). Addressing these with simulations is hindered by their massive scale, both in synthetic data volumes and numerical complexity, while real-data efforts are constrained by models that inadequately reflect the Earth's complexity and by sparse sensor measurements from the field. Recent machine learning (ML) efforts offer promise, but progress is obscured by a lack of proper characterization, fair reporting, and rigorous comparisons. To address this, we introduce a Common Task Framework (CTF) for ML for seismic wavefields, starting with three distinct wavefield datasets. Our CTF features a curated set of datasets at various scales (global, crustal, and local) and task-specific metrics spanning forecasting, reconstruction, and generalization under realistic constraints such as noise and limited data. Inspired by CTFs in fields like natural language processing, this framework provides a structured and rigorous foundation for head-to-head algorithm evaluation. We illustrate the evaluation procedure with scores reported for two of the datasets, showcasing the performance of various methods and foundation models for reconstructing seismic wavefields from both simulated and real-world sensor measurements. The CTF scores reveal the strengths, limitations, and suitability for specific problem classes. Our vision is to replace ad hoc comparisons with standardized evaluations on hidden test sets, raising the bar for rigor and reproducibility in scientific ML.

---

## 3. Modeling Non-Ergodic Path Effects Using Conditional Generative Model for Fourier Amplitude Spectra

**论文链接:** [http://arxiv.org/abs/2512.19909v1](http://arxiv.org/abs/2512.19909v1)

**作者:** Maxime Lacour, Pu Ren, Rie Nakata, Nori Nakata, Michael Mahoney

**发布时间:** 2025-12-22

### GPT解析

### 总结

研究提出了一种名为CGM-FAS的深度学习方法，作为基于高斯过程的非平稳地震动模型的替代方案，用于模拟傅里叶振幅谱中的非平稳路径效应。该方法通过条件变分自编码器架构直接从数据中学习空间模式和频率间相关性，无需预设相关函数，且计算效率高。

### 背景

非平稳地震动模型(GMMs)能够明确模拟源、场地和路径效应中的系统性空间变化，将标准差降低到平稳模型的30-40%，使更准确的场地特定地震危险性分析成为可能。然而，当前基于高斯过程(GP)的非平稳GMM方法存在计算限制，难以应用于大规模预测。

### 目的

提出一种深度学习方法(CGM-FAS)作为基于GP的方法的替代方案，用于模拟傅里叶振幅谱(FAS)中的非平稳路径效应，解决现有方法在大规模预测中的计算效率问题。

### 方法

CGM-FAS使用条件变分自编码器架构，通过使用地震和站点的地理坐标作为条件变量，直接从数据中学习空间模式和频率间相关性。研究使用旧金山湾区的地震数据对模型进行验证。

### 主要发现

CGM-FAS能够一致地预测非平稳路径效应，相比基于GP的方法具有三大优势：无需预设相关函数即可学习空间模式；能够捕获频率间相关性；计算效率高，可在几GB内存的情况下，10秒内为1,000个频率上的10,000个站点生成预测地图。

### 结论

这项工作展示了在多个频率和大空间域内高效进行非平稳地震动预测的有前景的方向，为地震危险性分析提供了新的可能性。

### 翻译

近期非平稳地震动模型(GMMs)的发展明确模拟了源、场地和路径效应中的系统性空间变化，将标准差降低到平稳模型的30-40%，使得更准确的场地特定地震危险性分析成为可能。当前非平稳GMM依赖于具有预设相关函数的高斯过程(GP)方法，因此在大规模预测中存在计算限制。本研究提出了一种名为条件生成傅里叶振幅谱模型(CGM-FAS)的深度学习方法，作为基于GP的方法的替代方案，用于模拟傅里叶振幅谱(FAS)中的非平稳路径效应。CGM-FAS使用条件变分自编码器架构，通过使用地震和站点的地理坐标作为条件变量，直接从数据中学习空间模式和频率间相关性。使用旧金山湾区地震数据，我们将CGM-FAS与该地区最近的基于GP的GMM进行比较，证明了CGM-FAS能够一致地预测非平稳路径效应。此外，与基于GP的方法相比，CGM-FAS在学习空间模式时无需预设相关函数，能够捕获频率间相关性，并能实现快速预测，在几GB内存的情况下，10秒内即可为1,000个频率上的10,000个站点生成地图。CGM-FAS的超参数可以调整，以确保生成的路径效应表现出与基于GP的经验GMM一致的变异性。这项工作展示了在多个频率和大空间域内高效进行非平稳地震动预测的有前景的方向。


### 论文摘要

Recent developments in non-ergodic ground-motion models (GMMs) explicitly model systematic spatial variations in source, site, and path effects, reducing standard deviation to 30-40% of ergodic models and enabling more accurate site-specific seismic hazard analysis. Current non-ergodic GMMs rely on Gaussian Process (GP) methods with prescribed correlation functions and thus have computational limitations for large-scale predictions. This study proposes a deep-learning approach called Conditional Generative Modeling for Fourier Amplitude Spectra (CGM-FAS) as an alternative to GP-based methods for modeling non-ergodic path effects in Fourier Amplitude Spectra (FAS). CGM-FAS uses a Conditional Variational Autoencoder architecture to learn spatial patterns and interfrequency correlation directly from data by using geographical coordinates of earthquakes and stations as conditional variables. Using San Francisco Bay Area earthquake data, we compare CGM-FAS against a recent GP-based GMM for the region and demonstrate consistent predictions of non-ergodic path effects. Additionally, CGM-FAS offers advantages compared to GP-based approaches in learning spatial patterns without prescribed correlation functions, capturing interfrequency correlations, and enabling rapid predictions, generating maps for 10,000 sites across 1,000 frequencies within 10 seconds using a few GB of memory. CGM-FAS hyperparameters can be tuned to ensure generated path effects exhibit variability consistent with the GP-based empirical GMM. This work demonstrates a promising direction for efficient non-ergodic ground-motion prediction across multiple frequencies and large spatial domains.

---

## 4. 论文ID: 2512.20557v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.20557v1.json'

---

## 5. 论文ID: 2512.20407v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.20407v1.json'

---

## 6. 论文ID: 2512.20128v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.20128v1.json'

---

## 7. Memory-T1: Reinforcement Learning for Temporal Reasoning in Multi-session Agents

**论文链接:** [http://arxiv.org/abs/2512.20092v1](http://arxiv.org/abs/2512.20092v1)

**作者:** Yiming Du, Baojun Wang, Yifan Xiang, Zhaowei Wang, Wenyu Huang, Boyang Xue, Bin Liang, Xingshan Zeng, Fei Mi, Haoli Bai, Lifeng Shang, Jeff Z. Pan, Yuxin Jiang, Kam-Fai Wong

**发布时间:** 2025-12-23

### GPT解析

### 总结

Memory-T1是一个使用强化学习学习时间感知记忆选择策略的框架，通过粗到细策略从长对话历史中选择相关证据，解决了长对话中时间推理困难的问题。

### 背景

在长多轮对话中进行时间推理是对话代理的关键能力，但现有研究表明，随着对话历史长度增加和噪声积累，当前长上下文模型难以准确识别时间相关信息，严重损害推理性能。

### 目的

开发一个框架来解决长对话中的时间推理问题，提高模型在长对话历史中识别时间相关信息的能力。

### 方法

Memory-T1框架使用强化学习学习时间感知的记忆选择策略，采用粗到细策略，先使用时间和相关性过滤器将对话历史剪枝为候选集，然后由RL代理选择精确证据会话。RL训练由多级奖励函数指导，优化答案准确性、证据基础和时间一致性，时间一致性奖励通过评估与查询时间范围的alignment解决时间歧义。

### 主要发现

在Time-Dialog基准测试上，Memory-T1将7B模型性能提升到67.0%，超过14B基线模型10.2%；消融研究表明时间一致性和证据基础奖励共同贡献15.0%性能提升；Memory-T1在高达128k tokens情况下保持鲁棒性，而基线模型崩溃。

### 结论

Memory-T1有效解决了长对话中的时间推理问题，在开源模型中建立了新的最先进性能，代码和数据集已公开可用。

### 翻译

在长多轮对话中进行时间推理是对话代理的关键能力。然而，现有工作和我们的初步研究表明，随着对话历史长度增加和噪声积累，当前长上下文模型难以准确识别时间相关信息，严重损害推理性能。为此，我们引入了Memory-T1，这是一个使用强化学习学习时间感知记忆选择策略的框架。它采用粗到细策略，首先使用时间和相关性过滤器将对话历史剪枝为候选集，然后由RL代理选择精确的证据会话。RL训练由多级奖励函数指导，优化(i)答案准确性，(ii)证据基础，和(iii)时间一致性。特别是，时间一致性奖励通过在会话级别（时间接近度）和话语级别（时间保真度）评估与查询时间范围的alignment，提供密集信号，使代理能够解决细微的时间歧义。在Time-Dialog基准测试上，Memory-T1将7B模型提升到总体得分67.0%，为开源模型建立了新的最先进性能，并超过14B基线模型10.2%。消融研究表明时间一致性和证据基础奖励共同贡献15.0%的性能提升。此外，Memory-T1在高达128k tokens的情况下保持鲁棒性，而基线模型崩溃，证明了其在广泛对话历史中对噪声的有效性。代码和数据集可在https://github.com/Elvin-Yiming-Du/Memory-T1/公开获取。


### 论文摘要

Temporal reasoning over long, multi-session dialogues is a critical capability for conversational agents. However, existing works and our pilot study have shown that as dialogue histories grow in length and accumulate noise, current long-context models struggle to accurately identify temporally pertinent information, significantly impairing reasoning performance. To address this, we introduce Memory-T1, a framework that learns a time-aware memory selection policy using reinforcement learning (RL). It employs a coarse-to-fine strategy, first pruning the dialogue history into a candidate set using temporal and relevance filters, followed by an RL agent that selects the precise evidence sessions. The RL training is guided by a multi-level reward function optimizing (i) answer accuracy, (ii) evidence grounding, and (iii) temporal consistency. In particular, the temporal consistency reward provides a dense signal by evaluating alignment with the query time scope at both the session-level (chronological proximity) and the utterance-level (chronological fidelity), enabling the agent to resolve subtle chronological ambiguities. On the Time-Dialog benchmark, Memory-T1 boosts a 7B model to an overall score of 67.0\%, establishing a new state-of-the-art performance for open-source models and outperforming a 14B baseline by 10.2\%. Ablation studies show temporal consistency and evidence grounding rewards jointly contribute to a 15.0\% performance gain. Moreover, Memory-T1 maintains robustness up to 128k tokens, where baseline models collapse, proving effectiveness against noise in extensive dialogue histories. The code and datasets are publicly available at https://github.com/Elvin-Yiming-Du/Memory-T1/

---

## 8. Skin Lesion Classification Using a Soft Voting Ensemble of Convolutional Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.20431v1](http://arxiv.org/abs/2512.20431v1)

**作者:** Abdullah Al Shafi, Abdul Muntakim, Pintu Chandra Shill, Rowzatul Zannat, Abdullah Al-Amin

**发布时间:** 2025-12-23

**DOI:** 10.1109/ECCE64574.2025.11013422

**备注:** Authors' version of the paper published in proceedings of ECCE, DOI: https://doi.org/10.1109/ECCE64574.2025.11013422

### GPT解析

### 总结

本文提出了一种使用CNN软投票集成进行早期皮肤癌分类的方法，通过混合双编码器进行分割，并结合MobileNetV2、VGG19和InceptionV3三个模型进行分类，在三个基准数据集上实现了高准确率。

### 背景

皮肤癌可以通过皮肤镜检查和眼科检查来识别，但早期检测可以显著提高生存机会。人工智能（AI）使用带注释的皮肤图像和卷积神经网络（CNNs）可以提高诊断准确性。

### 目的

开发一种使用CNN软投票集成进行早期皮肤癌分类的方法，提高诊断准确性和效率。

### 方法

使用三个基准数据集（HAM10000、ISIC 2016和ISIC 2019），应用重新平衡、图像增强和过滤技术，采用混合双编码器通过迁移学习进行分割，并通过MobileNetV2、VGG19和InceptionV3的集成进行分类，平衡准确性和速度以实现实际部署。

### 主要发现

在三个数据集上实现了病变识别准确率：96.32%（HAM10000）、90.86%（ISIC 2016）和93.92%（ISIC 2019），使用既定的皮肤病变检测指标评估系统性能，取得了令人印象深刻的结果。

### 结论

通过准确的分割，分类模型能够专注于临床显著特征，减少背景伪影，提高准确性，为皮肤癌的早期检测提供了有效工具。

### 翻译

皮肤癌可以通过皮肤镜检查和眼科检查来识别，但早期检测显著提高生存机会。人工智能（AI）使用带注释的皮肤图像和卷积神经网络（CNNs）提高诊断准确性。本文提出了一种使用CNN软投票集成进行早期皮肤癌分类的方法。本研究使用了三个基准数据集：HAM10000、ISIC 2016和ISIC 2019。过程涉及重新平衡、图像增强和过滤技术，随后通过迁移学习使用混合双编码器进行分割。准确的分割使分类模型能够专注于临床显著特征，减少背景伪影，提高准确性。分类通过MobileNetV2、VGG19和InceptionV3的集成进行，平衡准确性和速度以实现实际部署。该方法在三个数据集上实现了病变识别准确率：96.32%、90.86%和93.92%。系统性能使用既定的皮肤病变检测指标进行了评估，取得了令人印象深刻的结果。


### 论文摘要

Skin cancer can be identified by dermoscopic examination and ocular inspection, but early detection significantly increases survival chances. Artificial intelligence (AI), using annotated skin images and Convolutional Neural Networks (CNNs), improves diagnostic accuracy. This paper presents an early skin cancer classification method using a soft voting ensemble of CNNs. In this investigation, three benchmark datasets, namely HAM10000, ISIC 2016, and ISIC 2019, were used. The process involved rebalancing, image augmentation, and filtering techniques, followed by a hybrid dual encoder for segmentation via transfer learning. Accurate segmentation focused classification models on clinically significant features, reducing background artifacts and improving accuracy. Classification was performed through an ensemble of MobileNetV2, VGG19, and InceptionV3, balancing accuracy and speed for real-world deployment. The method achieved lesion recognition accuracies of 96.32\%, 90.86\%, and 93.92\% for the three datasets. The system performance was evaluated using established skin lesion detection metrics, yielding impressive results.

---

## 9. CLIP Based Region-Aware Feature Fusion for Automated BBPS Scoring in Colonoscopy Images

**论文链接:** [http://arxiv.org/abs/2512.20374v1](http://arxiv.org/abs/2512.20374v1)

**作者:** Yujia Fu, Zhiyu Dong, Tianwen Qian, Chenye Zheng, Danian Ji, Linhai Zhuo

**发布时间:** 2025-12-23

**备注:** 12 pages, 9 figures, BMVC 2025 submission

### GPT解析

### 总结

该研究提出了一种基于CLIP模型的自动化BBPS评分框架，通过融合全局视觉特征与粪便相关文本先验，提高了肠道清洁度评估的准确性，无需显式分割即可实现。

### 背景

准确的肠道清洁度评估对结肠镜检查至关重要，但现有的波士顿肠道准备量表(BBPS)评分系统在手动执行时存在主观性和观察者间变异性问题。

### 目的

开发一种自动化的BBPS评分框架，减少主观性并提高肠道清洁度评估的准确性。

### 方法

构建了一个包含2240张图像的高质量结肠镜数据集，来自517名受试者并标注了专家一致的BBPS评分；提出了一种新颖的自动化BBPS评分框架，利用基于适配器的迁移学习CLIP模型和专门的粪便特征提取分支，融合全局视觉特征与粪便相关文本先验。

### 主要发现

在自建数据集和公共NERHU数据集上的广泛实验表明，该方法优于现有基线，突显了其在计算机辅助结肠镜分析中的临床应用潜力。

### 结论

该方法能够更准确地评估肠道清洁度，有望用于临床实践，提高结肠镜检查的质量和效率。

### 翻译

准确的肠道清洁度评估对有效的结肠镜检查至关重要。波士顿肠道准备量表(BBPS)提供了一种标准化的评分系统，但在手动执行时存在主观性和观察者间变异性问题。在本文中，为了支持稳健的训练和评估，我们构建了一个高质量的结肠镜数据集，包含来自517名受试者的2240张图像，并标注了专家一致的BBPS评分。我们提出了一种新颖的自动化BBPS评分框架，利用基于适配器的迁移学习CLIP模型和专门的粪便特征提取分支。我们的方法融合了全局视觉特征与粪便相关的文本先验，提高了肠道清洁度评估的准确性，而无需显式分割。在我们数据集和公共NERHU数据集上的广泛实验证明了我们方法优于现有基线，突显了其在计算机辅助结肠镜分析中临床部署的潜力。


### 论文摘要

Accurate assessment of bowel cleanliness is essential for effective colonoscopy procedures. The Boston Bowel Preparation Scale (BBPS) offers a standardized scoring system but suffers from subjectivity and inter-observer variability when performed manually. In this paper, to support robust training and evaluation, we construct a high-quality colonoscopy dataset comprising 2,240 images from 517 subjects, annotated with expert-agreed BBPS scores. We propose a novel automated BBPS scoring framework that leverages the CLIP model with adapter-based transfer learning and a dedicated fecal-feature extraction branch. Our method fuses global visual features with stool-related textual priors to improve the accuracy of bowel cleanliness evaluation without requiring explicit segmentation. Extensive experiments on both our dataset and the public NERTHU dataset demonstrate the superiority of our approach over existing baselines, highlighting its potential for clinical deployment in computer-aided colonoscopy analysis.

---

## 10. 论文ID: 2512.16334v3

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.16334v3.json'

---

## 11. Gaussian Process Assisted Meta-learning for Image Classification and Object Detection Models

**论文链接:** [http://arxiv.org/abs/2512.20021v1](http://arxiv.org/abs/2512.20021v1)

**作者:** Anna R. Flowers, Christopher T. Franck, Robert B. Gramacy, Justin A. Krometis

**发布时间:** 2025-12-23

**备注:** 15 pages, 8 figures

### GPT解析

### 总结

本研究提出了一种通过利用元数据和计算机实验方法来指导机器学习数据采集的策略，以提高模型性能，特别是在代表性不足的条件下。

### 背景

收集操作上真实的数据来训练机器学习模型成本高昂，且在收集新数据前了解模型的不足之处非常重要。例如，在稀有物体图像上训练的目标检测器可能在代表性不足的条件下表现不佳。

### 目的

开发一种方法，通过利用描述训练数据收集背景的元数据（如季节、时间、地点）来指导后续数据采集，从而最大化模型性能。

### 方法

通过根据元数据变化训练数据来评估学习器，然后拟合一个高斯过程代理模型来响应这一变化，从而指导新的数据采集。这是一种元学习方法。

### 主要发现

这种元学习方法相比随机选择元数据的数据，可以提高学习器性能，这在经典学习示例和一个涉及收集航空图像以搜索飞机的实际应用中得到了验证。

### 结论

通过利用元数据和计算机实验工具包来指导数据采集，可以有效提高机器学习模型在代表性不足条件下的性能。

### 翻译

收集操作上真实的数据来支持机器学习模型可能成本高昂。在收集新数据之前，了解模型的不足之处是有帮助的。例如，在稀有物体图像上训练的目标检测器可能在代表性不足的条件下表现不佳。我们提出了一种方法，通过利用计算机实验工具包和描述训练数据收集背景的元数据（如季节、一天中的时间、位置）来指导后续的数据采集，以最大化模型性能。我们通过根据元数据变化训练数据来评估学习器，然后拟合一个高斯过程代理模型来响应这一变化，从而指导新的数据采集。这种元学习方法相比随机选择元数据的数据，可以提高学习器性能，作者在经典学习示例和一个涉及收集航空图像以搜索飞机的实际应用中证明了这一点。


### 论文摘要

Collecting operationally realistic data to inform machine learning models can be costly. Before collecting new data, it is helpful to understand where a model is deficient. For example, object detectors trained on images of rare objects may not be good at identification in poorly represented conditions. We offer a way of informing subsequent data acquisition to maximize model performance by leveraging the toolkit of computer experiments and metadata describing the circumstances under which the training data was collected (e.g., season, time of day, location). We do this by evaluating the learner as the training data is varied according to its metadata. A Gaussian process (GP) surrogate fit to that response surface can inform new data acquisitions. This meta-learning approach offers improvements to learner performance as compared to data with randomly selected metadata, which we illustrate on both classic learning examples, and on a motivating application involving the collection of aerial images in search of airplanes.

---

## 12. Regression of Functions by Quantum Neural Networks Circuits

**论文链接:** [http://arxiv.org/abs/2512.19978v1](http://arxiv.org/abs/2512.19978v1)

**作者:** Fernando M. de Paula Neto, Lucas dos Reis Silva, Paulo S. G. de Mattos Neto, Felipe F. Fanchini

**发布时间:** 2025-12-23

### GPT解析

### 总结

这篇论文提出了一种基于遗传算法的框架，用于自动构建量子神经网络架构，特别关注回归任务。研究发现量子模型可以在保持紧凑的同时提供与经典模型相当的性能，并且数据集复杂度指标可以可靠地预测最佳量子架构。

### 背景

量子神经网络模型的性能很大程度上取决于架构决策，包括电路深度、参数化操作的放置和数据编码策略。选择有效架构具有挑战性，且与经典神经网络拓扑选择这一计算难题密切相关。

### 目的

研究自动化量子电路构建方法，用于回归任务，引入一种遗传算法框架来发现简化的回归器量子神经网络架构，并分析数据集复杂度对量子架构选择的影响。

### 方法

提出一种遗传算法框架，探索电路深度、参数化门配置和灵活的数据重新上传模式，将量子回归器的构建表述为优化过程。在22个非线性基准函数和4个解析函数上与17个经典回归模型进行比较，使用12个结构描述符分析数据集复杂度，并在五个递增难度的元学习场景中测试这些度量。

### 主要发现

尽管经典方法通常能达到相似结果，但需要更多参数，而进化后的量子模型保持紧凑同时提供有竞争力的性能。数据集复杂度指标可以可靠地预测哪种量子架构表现最佳，在某些场景中表现出完美或接近完美的预测准确性。

### 结论

该研究为基于元学习的量子架构设计提供了理论基础，增进了对量子模型在回归设置中行为的理解，这些发现为更系统的、理论基础的量子回归方法铺平了道路。

### 翻译

量子神经网络模型的性能在很大程度上取决于架构决策，包括电路深度、参数化操作的放置和数据编码策略。选择有效架构具有挑战性，且与经典神经网络拓扑选择这一计算难题密切相关。本研究探讨了用于回归任务的自动化量子电路构建，并引入了一种遗传算法框架，用于发现简化的回归器量子神经网络架构。该方法探索深度、参数化门配置和灵活的数据重新上传模式，将量子回归器的构建表述为优化过程。在22个非线性基准函数和4个解析函数上评估发现的电路，并与17个经典回归模型进行比较。尽管经典方法通常能达到相似结果，但它们通常需要更多参数，而进化后的量子模型在保持紧凑的同时提供有竞争力的性能。我们进一步使用12个结构描述符分析数据集复杂度，并表明，在五个递增难度的元学习场景中，这些度量可以可靠地预测哪种量子架构将表现最佳。在几个场景中，结果表现出完美或接近完美的预测准确性，表明复杂度指标提供了数据集结构的强大而紧凑的表示，并可以有效地指导自动化模型选择。总体而言，本研究为基于元学习的量子架构设计提供了理论基础，并增进了对量子模型在回归设置中行为的理解——这一主题在先前工作中受到有限探索。这些发现为更系统和理论基础的量子回归方法铺平了道路。


### 论文摘要

The performance of quantum neural network models depends strongly on architectural decisions, including circuit depth, placement of parametrized operations, and data-encoding strategies. Selecting an effective architecture is challenging and closely related to the classical difficulty of choosing suitable neural-network topologies, which is computationally hard. This work investigates automated quantum-circuit construction for regression tasks and introduces a genetic-algorithm framework that discovers Reduced Regressor QNN architectures. The approach explores depth, parametrized gate configurations, and flexible data re-uploading patterns, formulating the construction of quantum regressors as an optimization process. The discovered circuits are evaluated against seventeen classical regression models on twenty-two nonlinear benchmark functions and four analytical functions. Although classical methods often achieve comparable results, they typically require far more parameters, whereas the evolved quantum models remain compact while providing competitive performance. We further analyze dataset complexity using twelve structural descriptors and show, across five increasingly challenging meta-learning scenarios, that these measures can reliably predict which quantum architecture will perform best. The results demonstrate perfect or near-perfect predictive accuracy in several scenarios, indicating that complexity metrics offer powerful and compact representations of dataset structure and can effectively guide automated model selection. Overall, this study provides a principled basis for meta-learning-driven quantum architecture design and advances the understanding of how quantum models behave in regression settings--a topic that has received limited exploration in prior work. These findings pave the way for more systematic and theoretically grounded approaches to quantum regression.

---

## 13. Fine-Tuned In-Context Learners for Efficient Adaptation

**论文链接:** [http://arxiv.org/abs/2512.19879v1](http://arxiv.org/abs/2512.19879v1)

**作者:** Jorg Bornschein, Clare Lyle, Yazhe Li, Amal Rannen-Triki, Xu Owen He, Razvan Pascanu

**发布时间:** 2025-12-22

### GPT解析

### 总结

研究了一种统一方法，将上下文学习直接整合到微调过程中，结合了两种主流大型语言模型适应方法的优点

### 背景

在将大型语言模型适应特定下游任务时，主要有两种方法：提示工程(利用模型固有泛化能力)和微调(直接优化模型参数)，但各自在数据量不同的情况下表现各异

### 目的

研究一种统一方法，将上下文学习和微调两种范式结合起来，克服各自在数据量不同情况下的局限性

### 方法

在特定任务数据上微调模型，这些数据通过添加上下文示例进行增强以模拟k-shot提示结构；使用预评估方法进行低数据量下的超参数选择，避免交叉验证同时利用所有可用数据

### 主要发现

统一方法虽然需要每个任务进行微调，但结合了上下文学习的样本效率和微调的性能提升，能够持续匹配并通常显著超过两种基线方法

### 结论

通过广泛实证研究比较了微调、上下文学习和统一方法在具体数据下游任务上的预测性能

### 翻译

当将大型语言模型适应特定下游任务时，通常采用两种主要方法：(1)提示工程，通常结合上下文少样本学习，利用模型的固有泛化能力；(2)在特定任务数据上进行微调，直接优化模型参数。虽然基于提示的方法在少样本场景下表现出色，但随着数据增加，其效果往往趋于平稳。相反，微调方法能很好地随数据扩展，但在训练样本稀少时可能表现不佳。我们研究了一种统一方法，通过将上下文学习直接整合到微调过程中来桥接这两种范式。具体来说，我们在添加了上下文示例的特定任务数据上微调模型，模拟k-shot提示的结构。这种方法虽然需要每个任务进行微调，但结合了上下文学习的样本效率和微调的性能提升，使其能够持续匹配并通常显著超过这两种基线方法。为了在低数据量下进行超参数选择，我们提出使用预评估方法，消除了昂贵的交叉验证需求，同时利用所有可用数据进行训练并提供稳健的验证信号。我们进行了广泛的实证研究，以确定哪种适应范式——微调、上下文学习或我们提出的统一方法——在具体数据的下游任务上提供最佳的预测性能。


### 论文摘要

When adapting large language models (LLMs) to a specific downstream task, two primary approaches are commonly employed: (1) prompt engineering, often with in-context few-shot learning, leveraging the model's inherent generalization abilities, and (2) fine-tuning on task-specific data, directly optimizing the model's parameters. While prompt-based methods excel in few-shot scenarios, their effectiveness often plateaus as more data becomes available. Conversely, fine-tuning scales well with data but may underperform when training examples are scarce. We investigate a unified approach that bridges these two paradigms by incorporating in-context learning directly into the fine-tuning process. Specifically, we fine-tune the model on task-specific data augmented with in-context examples, mimicking the structure of k-shot prompts. This approach, while requiring per-task fine-tuning, combines the sample efficiency of in-context learning with the performance gains of fine-tuning, leading to a method that consistently matches and often significantly exceeds both these baselines. To perform hyperparameter selection in the low-data regime, we propose to use prequential evaluation, which eliminates the need for expensive cross-validation and leverages all available data for training while simultaneously providing a robust validation signal. We conduct an extensive empirical study to determine which adaptation paradigm - fine-tuning, in-context learning, or our proposed unified approach offers the best predictive performance on a concrete data downstream-tasks.

---

## 14. 论文ID: 2512.19744v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19744v1.json'

---

## 15. 论文ID: 2512.20605v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.20605v1.json'

---

## 16. 论文ID: 2512.20424v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.20424v1.json'

---

## 17. AMoE: Agglomerative Mixture-of-Experts Vision Foundation Model

**论文链接:** [http://arxiv.org/abs/2512.20157v1](http://arxiv.org/abs/2512.20157v1)

**作者:** Sofian Chaybouti, Sanath Narayan, Yasser Dahou, Phúc H. Lê Khac, Ankit Singh, Ngoc Dung Huynh, Wamiq Reyaz Para, Hilde Kuehne, Hakim Hacid

**发布时间:** 2025-12-23

**备注:** 17 pages, 8 figures, 11 tables

### GPT解析

### 总结

本研究系统性地研究了视觉基础模型的多教师蒸馏方法，提出了AMoE模型，并确定了三种关键因素来提高训练效率，同时创建了OpenLVD200M数据集并发布了相关模型。

### 背景

通过多教师蒸馏训练的视觉基础模型为统一的视觉表示提供了有前途的路径，但这类方法的学习动态性和数据效率仍然未被充分探索。

### 目的

系统地研究视觉基础模型的多教师蒸馏，并确定能够以较低计算成本进行训练的关键因素。

### 方法

引入Agglomerative Mixture-of-Experts视觉基础模型(AMoE)，同时将SigLIP2和DINOv3的知识蒸馏到Mixture-of-Experts学生模型中。

### 主要发现

1) 非对称关系知识蒸馏损失函数保留了每个教师的几何特性，同时实现有效的知识转移；2) 令牌平衡批处理将不同分辨率的图像打包成具有统一令牌预算的序列，稳定了不同分辨率的表示学习；3) 训练数据的层次聚类和采样相比随机采样显著提高了多教师蒸馏的样本效率。

### 结论

通过结合这些发现，整理了OpenLVD200M，一个2亿图像语料库，展示了多教师蒸馏的卓越效率，并在Mixture-of-Experts架构中实例化，同时发布了OpenLVD200M数据集和蒸馏模型。

### 翻译

通过多教师蒸馏训练的视觉基础模型为统一的视觉表示提供了一条有前途的路径，但此类方法的学习动态性和数据效率仍然未被充分探索。在本文中，我们系统性地研究了视觉基础模型的多教师蒸馏，并确定了能够以较低计算成本进行训练的关键因素。我们引入了聚合式专家混合视觉基础模型(AMoE)，该模型同时将SigLIP2和DINOv3的知识蒸馏到专家混合学生模型中。我们证明：(1)我们的非对称关系知识蒸馏损失函数保留了每个教师的几何特性，同时实现了有效的知识转移；(2)令牌平衡批处理将不同分辨率的图像打包成具有统一令牌预算的序列，在不牺牲性能的情况下稳定了不同分辨率的表示学习；(3)训练数据的层次聚类和采样（通常保留给自监督学习）相比随机采样显著提高了多教师蒸馏的样本效率。通过结合这些发现，我们整理了OpenLVD200M，一个2亿图像语料库，展示了多教师蒸馏的卓越效率。在专家混合架构中实例化。我们发布了OpenLVD200M和蒸馏模型。


### 论文摘要

Vision foundation models trained via multi-teacher distillation offer a promising path toward unified visual representations, yet the learning dynamics and data efficiency of such approaches remain underexplored. In this paper, we systematically study multi-teacher distillation for vision foundation models and identify key factors that enable training at lower computational cost. We introduce Agglomerative Mixture-of-Experts Vision Foundation Models (AMoE), which distill knowledge from SigLIP2 and DINOv3 simultaneously into a Mixture-of-Experts student. We show that (1) our Asymmetric Relation-Knowledge Distillation loss preserves the geometric properties of each teacher while enabling effective knowledge transfer, (2) token-balanced batching that packs varying-resolution images into sequences with uniform token budgets stabilizes representation learning across resolutions without sacrificing performance, and (3) hierarchical clustering and sampling of training data--typically reserved for self-supervised learning--substantially improves sample efficiency over random sampling for multi-teacher distillation. By combining these findings, we curate OpenLVD200M, a 200M-image corpus that demonstrates superior efficiency for multi-teacher distillation. Instantiated in a Mixture-of-Experts. We release OpenLVD200M and distilled models.

---

## 18. Retrieval-augmented Prompt Learning for Pre-trained Foundation Models

**论文链接:** [http://arxiv.org/abs/2512.20145v1](http://arxiv.org/abs/2512.20145v1)

**作者:** Xiang Chen, Yixin Ou, Quan Feng, Lei Li, Piji Li, Haibo Ye, Sheng-Jun Huang, Shuofei Qiao, Shumin Deng, Huajun Chen, Ningyu Zhang

**发布时间:** 2025-12-23

**DOI:** 10.1109/TASLPRO.2025.3608936

**备注:** IEEE/ACM Transactions on Audio, Speech and Language Processing

### GPT解析

### 总结

RetroPrompt是一种新型提示学习方法，通过利用知识库和检索机制，解决了传统提示学习中记忆与泛化平衡的问题，在零样本和少样本场景下表现优越。

### 背景

预训练基础模型已成为促进大规模多模态学习的关键工具。研究人员通过提示学习采用'预训练、提示、预测'范式改进少样本性能，但传统提示学习方法遵循参数化学习范式，可能导致记忆和机械学习的泛化稳定性受影响，且在充分利用非典型实例和避免有限数据下的过拟合方面存在困难。

### 目的

克服传统提示学习的局限性，实现记忆和泛化之间的平衡，将知识与单纯的记忆解耦。

### 方法

提出名为RetroPrompt的方法，利用从训练数据生成的公开可访问的知识库，在输入、训练和推理阶段都纳入检索机制，使模型能够主动从语料库中检索相关的上下文信息，增强可用线索。

### 主要发现

在自然语言处理和计算机视觉任务的各种数据集上进行了全面实验，RetroPrompt在零样本和少样本场景下表现出优越性能。通过对记忆模式的详细分析，观察到RetroPrompt有效减少了对机械记忆的依赖，导致泛化能力增强。

### 结论

RetroPrompt通过解耦知识与单纯记忆，有效平衡了记忆和泛化，减少了模型对机械记忆的依赖，从而提高了泛化能力。

### 翻译

预训练基础模型已成为促进大规模多模态学习的关键工具。研究人员通过提示学习有效采用'预训练、提示、预测'范式，改进了少样本性能。然而，针对PFMs的提示学习方法仍然遵循参数化学习范式。因此，记忆和机械学习中的泛化稳定性可能会受到影响。更具体地说，传统提示学习在充分利用非典型实例和避免在有限数据下的完全监督训练过程中对浅层模式的过拟合方面可能面临困难。为了克服这些限制，我们提出了名为RetroPrompt的方法，旨在通过将知识与单纯记忆解耦来实现记忆与泛化之间的平衡。与传统提示方法不同，RetroPrompt利用从训练数据生成的公开可访问的知识库，并在输入、训练和推理阶段都纳入了检索机制。这使得模型能够主动从语料库中检索相关的上下文信息，从而增强可用线索。我们在自然语言处理和计算机视觉任务的各种数据集上进行了全面实验，证明了我们提出的方法RetroPrompt在零样本和少样本场景下的优越性能。通过对记忆模式的详细分析，我们观察到RetroPrompt有效减少了对机械记忆的依赖，从而提高了泛化能力。


### 论文摘要

The pre-trained foundation models (PFMs) have become essential for facilitating large-scale multimodal learning. Researchers have effectively employed the ``pre-train, prompt, and predict'' paradigm through prompt learning to induce improved few-shot performance. However, prompt learning approaches for PFMs still follow a parametric learning paradigm. As such, the stability of generalization in memorization and rote learning can be compromised. More specifically, conventional prompt learning might face difficulties in fully utilizing atypical instances and avoiding overfitting to shallow patterns with limited data during the process of fully-supervised training. To overcome these constraints, we present our approach, named RetroPrompt, which aims to achieve a balance between memorization and generalization by decoupling knowledge from mere memorization. Unlike traditional prompting methods, RetroPrompt leverages a publicly accessible knowledge base generated from the training data and incorporates a retrieval mechanism throughout the input, training, and inference stages. This enables the model to actively retrieve relevant contextual information from the corpus, thereby enhancing the available cues. We conduct comprehensive experiments on a variety of datasets across natural language processing and computer vision tasks to demonstrate the superior performance of our proposed approach, RetroPrompt, in both zero-shot and few-shot scenarios. Through detailed analysis of memorization patterns, we observe that RetroPrompt effectively reduces the reliance on rote memorization, leading to enhanced generalization.

---

## 19. Reason2Decide: Rationale-Driven Multi-Task Learning

**论文链接:** [http://arxiv.org/abs/2512.20074v1](http://arxiv.org/abs/2512.20074v1)

**作者:** H M Quamran Hasan, Housam Khalifa Bashier, Jiayi Dai, Mi-Young Kim, Randy Goebel

**发布时间:** 2025-12-23

### GPT解析

### 总结

Reason2Decide是一个创新的两阶段训练框架，解决了临床决策支持系统中预测准确性和解释一致性的关键挑战。该框架通过处理暴露偏差和任务分离问题，在各种医疗数据集上表现出色，同时使用比当代基础模型小40倍的模型实现高性能，使其成为资源受限环境中的理想解决方案。

### 背景

大型语言模型已被广泛采用，但临床决策支持系统面临关键挑战：在生成与预测一致的解释的同时实现高预测准确性。当前方法存在暴露偏差问题，导致解释与预测不一致。

### 目的

解决自理性化中的关键挑战，包括暴露偏差和任务分离，提出一个两阶段训练框架，能够同时提高预测准确性和解释一致性。

### 方法

提出了名为Reason2Decide的两阶段训练框架：第一阶段在推理生成上训练模型；第二阶段联合训练标签预测和推理生成，使用计划采样技术，逐渐从基于真实标签的条件化过渡到基于模型预测的条件化。在三个医疗数据集上评估，包括专用的分诊数据集和公开的生物医学问答数据集。

### 主要发现

在不同模型规模下，Reason2Decide在预测（F1）和推理保真度方面优于其他微调基线和一些零样本LLM；在分诊任务中，对多种推理源具有鲁棒性；仅使用LLM生成的推理进行预训练就优于其他微调变体，表明减少了对人工标注的依赖；使用比当代基础模型小40倍的模型实现了这些提升。

### 结论

Reason2Decide使临床推理在资源受限部署中更易于访问，同时仍提供可解释的决策支持。该框架能够在保持高性能的同时，大幅减少模型大小，提高临床决策支持系统的实用性。

### 翻译

尽管大型语言模型被广泛采用，临床决策支持系统仍面临一个关键挑战：在生成与预测一致解释的同时实现高预测准确性。当前方法因存在暴露偏差而导致解释与预测不一致。我们提出了Reason2Decide，一个两阶段训练框架，解决了自理性化中的关键挑战，包括暴露偏差和任务分离。在第一阶段，我们的模型在推理生成上训练；在第二阶段，我们联合训练标签预测和推理生成，应用计划采样技术，逐渐从基于真实标签的条件化过渡到基于模型预测的条件化。我们在三个医疗数据集上评估Reason2Decide，包括专用的分诊数据集和公开的生物医学问答数据集。在不同模型规模下，Reason2Decide在预测（F1）和推理保真度方面优于其他微调基线和一些零样本LLM。在分诊任务中，Reason2Decide对LLM生成、护士编写和护士后处理的推理具有推理源鲁棒性。在我们的实验中，尽管仅在第一阶段使用LLM生成的推理，Reason2Decide仍优于其他微调变体。这表明LLM生成的推理适合用于模型预训练，减少对人工标注的依赖。值得注意的是，Reason2Decide使用比当代基础模型小40倍的模型实现了这些提升，使临床推理在资源受限部署中更易于访问，同时仍提供可解释的决策支持。


### 论文摘要

Despite the wide adoption of Large Language Models (LLM)s, clinical decision support systems face a critical challenge: achieving high predictive accuracy while generating explanations aligned with the predictions. Current approaches suffer from exposure bias leading to misaligned explanations. We propose Reason2Decide, a two-stage training framework that addresses key challenges in self-rationalization, including exposure bias and task separation. In Stage-1, our model is trained on rationale generation, while in Stage-2, we jointly train on label prediction and rationale generation, applying scheduled sampling to gradually transition from conditioning on gold labels to model predictions. We evaluate Reason2Decide on three medical datasets, including a proprietary triage dataset and public biomedical QA datasets. Across model sizes, Reason2Decide outperforms other fine-tuning baselines and some zero-shot LLMs in prediction (F1) and rationale fidelity (BERTScore, BLEU, LLM-as-a-Judge). In triage, Reason2Decide is rationale source-robust across LLM-generated, nurse-authored, and nurse-post-processed rationales. In our experiments, while using only LLM-generated rationales in Stage-1, Reason2Decide outperforms other fine-tuning variants. This indicates that LLM-generated rationales are suitable for pretraining models, reducing reliance on human annotations. Remarkably, Reason2Decide achieves these gains with models 40x smaller than contemporary foundation models, making clinical reasoning more accessible for resource-constrained deployments while still providing explainable decision support.

---

## 20. WSD-MIL: Window Scale Decay Multiple Instance Learning for Whole Slide Image Classification

**论文链接:** [http://arxiv.org/abs/2512.19982v1](http://arxiv.org/abs/2512.19982v1)

**作者:** Le Feng, Li Xiao

**发布时间:** 2025-12-23

### GPT解析

### 总结

本文提出了一种名为窗口尺度衰减多实例学习(WSD-MIL)的新方法，用于解决计算病理学中全切片图像分析的问题。该方法通过窗口尺度衰减注意力和挤压-激励区域门控模块，有效处理了不同尺度肿瘤区域的建模，同时提高了计算效率。

### 背景

近年来，预训练基础模型与多实例学习(MIL)的结合提高了计算病理学中的诊断准确性。然而，现有MIL方法专注于优化特征提取器和聚合策略，忽略了全切片图像中实例间的复杂语义关系。基于Transformer的MIL方法虽能建模实例依赖关系，但其二次计算复杂度限制了在大规模WSI上的应用。此外，不同WSI中肿瘤区域规模的显著变化使得固定尺度注意力方法难以精确捕获局部实例相关性，也无法考虑补丁相关性的距离衰减效应。

### 目的

开发一种能够增强对变尺度肿瘤区域建模能力同时提高计算效率的方法，解决现有基于Transformer的MIL方法的局限性。

### 方法

提出窗口尺度衰减多实例学习(WSD-MIL)，包含两个主要模块：1)基于窗口尺度衰减的注意力模块，采用基于聚类的采样策略降低计算成本，同时逐步衰减注意力窗口尺度以捕获不同尺度的局部实例关系；2)基于挤压和激励的区域门控模块，动态调整窗口权重以增强全局信息建模。

### 主要发现

WSD-MIL在CAMELYON16和TCGA-BRCA数据集上实现了最先进的性能，同时减少了62%的计算内存消耗。

### 结论

WSD-MIL有效解决了现有方法在处理变尺度肿瘤区域和计算效率方面的问题，为计算病理学中的全切片图像分析提供了新的解决方案。代码将公开可用。

### 翻译

近年来，预训练基础模型与多实例学习(MIL)的结合提高了计算病理学中的诊断准确性。然而，现有的MIL方法专注于优化特征提取器和聚合策略，而忽略了全切片图像(WSI)中实例之间的复杂语义关系。尽管基于Transformer的MIL方法旨在建模实例依赖关系，但二次计算复杂度限制了它们在大规模WSI上的可扩展性。此外，由于不同WSI中肿瘤区域规模的显著变化，现有基于Transformer的方法采用固定尺度注意力机制，在精确捕获局部实例相关性方面面临重大挑战，并且无法考虑补丁相关性的距离衰减效应。为解决这些挑战，我们提出了窗口尺度衰减MIL(WSD-MIL)，旨在增强对变尺度肿瘤区域的建模能力同时提高计算效率。WSD-MIL包括：1)基于窗口尺度衰减的注意力模块，采用基于聚类的采样策略降低计算成本，同时逐步衰减注意力窗口尺度以捕获不同尺度的局部实例关系；以及2)基于挤压和激励的区域门控模块，动态调整窗口权重以增强全局信息建模。实验结果表明，WSD-MIL在CAMELYON16和TCGA-BRCA数据集上实现了最先进的性能，同时减少了62%的计算内存。代码将公开可用。


### 论文摘要

In recent years, the integration of pre-trained foundational models with multiple instance learning (MIL) has improved diagnostic accuracy in computational pathology. However, existing MIL methods focus on optimizing feature extractors and aggregation strategies while overlooking the complex semantic relationships among instances within whole slide image (WSI). Although Transformer-based MIL approaches aiming to model instance dependencies, the quadratic computational complexity limits their scalability to large-scale WSIs. Moreover, due to the pronounced variations in tumor region scales across different WSIs, existing Transformer-based methods employing fixed-scale attention mechanisms face significant challenges in precisely capturing local instance correlations and fail to account for the distance-based decay effect of patch relevance. To address these challenges, we propose window scale decay MIL (WSD-MIL), designed to enhance the capacity to model tumor regions of varying scales while improving computational efficiency. WSD-MIL comprises: 1) a window scale decay based attention module, which employs a cluster-based sampling strategy to reduce computational costs while progressively decaying attention window-scale to capture local instance relationships at varying scales; and 2) a squeeze-and-excitation based region gate module, which dynamically adjusts window weights to enhance global information modeling. Experimental results demonstrate that WSD-MIL achieves state-of-the-art performance on the CAMELYON16 and TCGA-BRCA datasets while reducing 62% of the computational memory. The code will be publicly available.

---

## 21. How Much 3D Do Video Foundation Models Encode?

**论文链接:** [http://arxiv.org/abs/2512.19949v1](http://arxiv.org/abs/2512.19949v1)

**作者:** Zixuan Huang, Xiang Li, Zhaoyang Lv, James M. Rehg

**发布时间:** 2025-12-23

**备注:** Project Page: https://vidfm-3d-probe.github.io

### GPT解析

### 总结

研究视频基础模型的3D理解能力

### 背景

视频是3D世界的连续2D投影，在大型视频数据上训练后是否会自然产生全局3D理解尚不明确

### 目的

评估现有视频基础模型(VidFMs)的3D理解能力

### 方法

提出第一个模型无关的框架，通过从特征中估计多个3D属性来测量各种VidFMs的3D意识

### 主要发现

最先进的视频生成模型表现出对3D物体和场景的强烈理解，尽管没有在3D数据上训练，这种理解甚至超过专门为3D任务训练的大型专家模型

### 结论

主要VidFMs的3D基准测试为构建可扩展的3D模型提供了有价值的观察

### 翻译

视频是3D世界的连续2D投影。在大型视频数据上训练后，全局3D理解是否会自然出现？我们通过量化现有视频基础模型(VidFMs)的3D理解能力来研究这一问题，这些模型在大量视频数据上进行了预训练。我们提出了第一个模型无关的框架，通过从特征中估计多个3D属性来测量各种VidFMs的3D意识。我们的研究在多个轴上提供了关于VidFMs的3D意识的有意义发现。特别是，我们表明最先进的视频生成模型表现出对3D物体和场景的强烈理解，尽管它们没有在3D数据上进行训练。这种理解甚至可以超过专门为3D任务训练的大型专家模型。我们的发现以及主要VidFMs的3D基准测试，为构建可扩展的3D模型提供了有价值的观察。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文研究视频基础模型（VidFMs）在多大程度上编码了3D理解能力。这个问题重要是因为视频是3D世界的2D投影，更容易获取大规模数据，而3D数据获取困难且昂贵。理解视频模型是否具有内在的3D理解能力有助于开发可扩展的3D世界模型，对AR/VR和具身AI等领域有广泛应用价值，同时可以确定是否需要额外3D数据来训练强大的3D模型。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到从2D视觉恢复3D结构是计算机视觉中的长期问题，3D数据获取困难限制了数据驱动方法发展。他们观察到视频数据更容易获取且具有多样性，提出直接评估视频模型3D理解能力的需求。设计方法时借鉴了Probe3D和Feat2GS等使用密集探测评估3D意识的工作，参考了VGGT的架构设计，并利用了现有的视频生成模型和自监督视频编码器作为基础模型，但提出了一个模型无关的框架，通过浅层读取模块直接估计3D属性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是：如果视频模型理解3D世界，应该能够通过前馈方式使用浅层读取模块提取准确的3D属性，无需对基础模型进行后优化或微调。整体流程包括：1)特征提取：从视频运行VidFM提取每帧空间特征，对扩散模型使用类似DIFT的方法；2)3D意识探测：使用浅层transformer架构，交替注意力和三个读取头（点图、深度图和相机姿态）；3)训练与评估：使用多任务目标函数训练探测模型，评估点图、姿态和深度预测误差作为3D意识指标。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个模型无关框架评估VidFMs的3D意识；2)通过浅层读取模块直接估计3D属性，而非使用间接指标；3)从多个维度（范围、影响因素、定位和实际应用）全面评估3D意识；4)发现前沿视频生成模型具有强大的3D理解能力，甚至可以媲美专门为3D任务训练的模型。相比之前工作，本文使用直接的前馈3D预测任务而非间接指标，评估了各种视频模型而非仅图像模型，从多维度评估3D意识，并发现了视频模型比预期更强的3D理解能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文首次提出了一种模型无关的框架评估视频基础模型的3D意识，发现前沿视频生成模型具有强大的3D理解能力，甚至可以媲美专门为3D任务训练的模型，为构建可扩展的3D世界模型提供了新见解。'}


### 论文摘要

Videos are continuous 2D projections of 3D worlds. After training on large video data, will global 3D understanding naturally emerge? We study this by quantifying the 3D understanding of existing Video Foundation Models (VidFMs) pretrained on vast video data. We propose the first model-agnostic framework that measures the 3D awareness of various VidFMs by estimating multiple 3D properties from their features via shallow read-outs. Our study presents meaningful findings regarding the 3D awareness of VidFMs on multiple axes. In particular, we show that state-of-the-art video generation models exhibit a strong understanding of 3D objects and scenes, despite not being trained on any 3D data. Such understanding can even surpass that of large expert models specifically trained for 3D tasks. Our findings, together with the 3D benchmarking of major VidFMs, provide valuable observations for building scalable 3D models.

---

## 22. How well do Large Language Models Recognize Instructional Moves? Establishing Baselines for Foundation Models in Educational Discourse

**论文链接:** [http://arxiv.org/abs/2512.19903v1](http://arxiv.org/abs/2512.19903v1)

**作者:** Kirk Vanacore, Rene F. Kizilcec

**发布时间:** 2025-12-22

### GPT解析

### 总结

本研究评估了六个大型语言模型在无需大量定制的情况下解释真实教育场景的能力，特别关注它们在分类课堂转录文本中教学动作方面的基本性能，并比较了不同提示方法的效果。

### 背景

大型语言模型越来越多地被应用于教育技术领域，用于生成教学材料、协助评估设计和辅导等任务。虽然之前的研究已经调查了如何调整或优化模型以适应特定任务，但对于LLMs在无需大量定制的情况下解释真实教育场景的能力了解甚少。

### 目的

随着基于LLM的系统在日常学术环境中被广泛采用，了解它们的开箱即用能力对于设定期望和基准测试变得越来越重要。本研究旨在估计LLMs在分类课堂转录文本中教学动作这一简单但重要任务上的基本性能。

### 方法

研究人员比较了六个大型语言模型，评估了三种典型的提示方法：零样本提示、单样本提示和少样本提示。他们在真实的课堂转录文本上测试了这些模型分类教学动作的能力。

### 主要发现

零样本性能表现中等；提供全面的示例（少样本提示）显著提高了最先进模型的性能，最强配置的Cohen's Kappa值达到了0.58（与专家编码注释相比）；改进既不均匀也不完整：性能因教学动作而异，较高的召回率经常以增加假正价为代价。

### 结论

总的来说，这些发现表明基础模型在解释教学话语方面具有有意义但有限的能力，提示设计有助于展现能力，但不能消除基本的可靠性限制。

### 翻译

大型语言模型(LLMs)越来越多地被应用于教育技术，用于各种任务，从生成教学材料和协助评估设计到辅导。虽然之前的研究已经调查了如何调整或优化模型以适应特定任务，但对于LLMs在无需大量定制的情况下解释真实教育场景的能力知之甚少。随着基于LLM的系统在日常学术环境中被学习者教育者广泛采用，了解它们的开箱即用能力对于设定期望和基准测试变得越来越重要。我们比较了六个LLMs，以估计它们在简单但重要的任务上的基本性能：分类真实课堂转录文本中的教学动作。我们评估了典型的提示方法：零样本、单样本和少样本提示。我们发现，虽然零样本性能中等，但提供全面的示例（少样本提示）显著提高了最先进模型的性能，最强配置的Cohen's Kappa值达到了0.58，与专家编码注释相当。与此同时，改进既不均匀也不完整：性能因教学动作而异，较高的召回率经常以增加假正价为代价。总的来说，这些发现表明基础模型在解释教学话语方面具有有意义但有限的能力，提示设计有助于展现能力，但不能消除基本的可靠性限制。


### 论文摘要

Large language models (LLMs) are increasingly adopted in educational technologies for a variety of tasks, from generating instructional materials and assisting with assessment design to tutoring. While prior work has investigated how models can be adapted or optimized for specific tasks, far less is known about how well LLMs perform at interpreting authentic educational scenarios without significant customization. As LLM-based systems become widely adopted by learners and educators in everyday academic contexts, understanding their out-of-the-box capabilities is increasingly important for setting expectations and benchmarking. We compared six LLMs to estimate their baseline performance on a simple but important task: classifying instructional moves in authentic classroom transcripts. We evaluated typical prompting methods: zero-shot, one-shot, and few-shot prompting. We found that while zero-shot performance was moderate, providing comprehensive examples (few-shot prompting) significantly improved performance for state-of-the-art models, with the strongest configuration reaching Cohen's Kappa = 0.58 against expert-coded annotations. At the same time, improvements were neither uniform nor complete: performance varied considerably by instructional move, and higher recall frequently came at the cost of increased false positives. Overall, these findings indicate that foundation models demonstrate meaningful yet limited capacity to interpret instructional discourse, with prompt design helping to surface capability but not eliminating fundamental reliability constraints.

---

## 23. LiDARDraft: Generating LiDAR Point Cloud from Versatile Inputs

**论文链接:** [http://arxiv.org/abs/2512.20105v1](http://arxiv.org/abs/2512.20105v1)

**作者:** Haiyun Wei, Fan Lu, Yunwei Zhu, Zehan Zheng, Weiyi Xue, Lin Shao, Xudong Zhang, Ya Wu, Rong Fu, Guang Chen

**发布时间:** 2025-12-23

### GPT解析

### 总结

本文提出了一种名为LiDARDraft的LiDAR点云生成方法，通过3D布局作为桥梁连接各种条件信号与LiDAR点云，实现了高质量且多样化的可控点云生成，能够从文本描述、图像和草图创建自动驾驶环境。

### 背景

生成真实且多样化的LiDAR点云对自动驾驶仿真至关重要，但现有方法在实现高质量结果的同时难以提供多样化的可控性，这源于LiDAR点云的复杂分布与简单控制信号之间的不平衡。

### 目的

解决现有方法的局限性，提出一种能够实现高质量结果和多样化可控性的LiDAR点云生成方法。

### 方法

提出LiDARDraft方法，利用3D布局作为各种条件信号与LiDAR点云之间的桥梁，将文本、图像和点云表示为统一的3D布局，并转换为语义和深度控制信号，采用基于rangemap的ControlNet指导点云生成。

### 主要发现

像素级对齐方法在可控LiDAR点云生成方面表现出色，能够实现'从零开始'的仿真，可以从任意文本描述、图像和草图创建自动驾驶环境。

### 结论

LiDARDraft方法成功解决了LiDAR点云复杂分布与简单控制信号之间的不平衡问题，实现了高质量且多样化的可控LiDAR点云生成。

### 翻译

生成真实且多样化的LiDAR点云对自动驾驶仿真至关重要。尽管先前的方法能够从用户输入生成LiDAR点云，但由于LiDAR点云的复杂分布与简单控制信号之间的不平衡，它们难以在获得高质量结果的同时实现多样化的可控性。为解决这一局限，我们提出了LiDARDraft，它利用3D布局在多种条件信号与LiDAR点云之间建立桥梁。3D布局可以从各种用户输入（如文本描述和图像）轻松生成。具体而言，我们将文本、图像和点云表示为统一的3D布局，进一步将其转换为语义和深度控制信号。然后，我们采用基于rangemap的ControlNet来指导LiDAR点云生成。这种像素级对齐方法在可控LiDAR点云生成方面表现出色，实现了'从零开始'的仿真，允许从任意的文本描述、图像和草图创建自动驾驶环境。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决如何从多样化输入（如文本、图像和点云）生成高质量激光雷达点云的问题。这在自动驾驶领域非常重要，因为自动驾驶系统需要大量数据进行训练和验证，而收集真实世界物理数据成本高昂、不安全且难以扩展。生成逼真的点云可以为自动驾驶提供丰富的训练数据，并实现'从零开始的仿真'，仅通过简单输入就能创建自动驾驶环境。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：早期VAE和GAN方法难以产生高质量结果；现有扩散模型在处理多样化输入时面临挑战；简单输入难以捕捉自动驾驶场景的复杂性。他们提出使用3D布局作为统一条件表示，桥接多样化输入与点云生成。该方法借鉴了扩散模型、ControlNet架构、语义分割、深度估计和大型语言模型等现有工作，但创新性地将它们组合用于点云生成任务。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用3D布局作为统一条件表示，连接多样化输入与点云生成。流程包括：1)输入处理：将文本、图像和点云转换为3D布局；2)场景射线投射：将布局投影为范围图像；3)布局引导的生成：使用基于范围图的ControlNet指导点云生成；4)训练：先训练无条件生成模型，再微调ControlNet实现条件生成。这种方法实现了像素级对齐的控制，确保生成结果既符合输入条件又保持多样性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一条件框架：支持文本、图像和点云三种输入模态；2)直接布局到点云控制：首次实现点级控制的布局到点云生成；3)高效训练：只需微调ControlNet，大幅降低训练成本。相比之前工作，LiDARDraft比LiDARGen提供更精细控制；比LiDARDM更真实可靠；比LiDARDiffusion提供更精确的像素级控制；比Text2LiDAR支持更多输入模态。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': "LiDARDraft提出了一种创新的统一框架，通过3D布局桥接多样化输入与激光雷达点云生成，实现了从文本、图像和点云生成高质量、可控的激光雷达点云，为自动驾驶仿真提供了'从零开始'的可能性。"}


### 论文摘要

Generating realistic and diverse LiDAR point clouds is crucial for autonomous driving simulation. Although previous methods achieve LiDAR point cloud generation from user inputs, they struggle to attain high-quality results while enabling versatile controllability, due to the imbalance between the complex distribution of LiDAR point clouds and the simple control signals. To address the limitation, we propose LiDARDraft, which utilizes the 3D layout to build a bridge between versatile conditional signals and LiDAR point clouds. The 3D layout can be trivially generated from various user inputs such as textual descriptions and images. Specifically, we represent text, images, and point clouds as unified 3D layouts, which are further transformed into semantic and depth control signals. Then, we employ a rangemap-based ControlNet to guide LiDAR point cloud generation. This pixel-level alignment approach demonstrates excellent performance in controllable LiDAR point clouds generation, enabling "simulation from scratch", allowing self-driving environments to be created from arbitrary textual descriptions, images and sketches.

---

## 24. Discovering Lie Groups with Flow Matching

**论文链接:** [http://arxiv.org/abs/2512.20043v1](http://arxiv.org/abs/2512.20043v1)

**作者:** Jung Yeon Park, Yuxuan Chen, Floor Eijkelboom, Jan-Willem van de Meent, Lawson L. S. Wong, Robin Walters

**发布时间:** 2025-12-23

### GPT解析

### 总结

本文提出了一种名为LieFlow的新方法，通过在李群上进行流匹配直接从数据中学习对称性，成功在2D和3D点云上发现了离散群，包括反射，并解决了对称排列导致的'最后时刻收敛'问题。

### 背景

对称性对于理解物理系统至关重要，同时也能改善机器学习中的性能和样本效率，两者都需要了解数据中潜在的对称性。

### 目的

开发一种直接从数据中学习对称性的方法，通过流匹配技术在李群上实现对称性的发现。

### 方法

将对称性发现表述为学习更大假设组上的分布，使学习分布匹配数据中观察到的对称性；提出LieFlow方法，相比先前工作在可发现的群类型上更灵活且需要更少假设。

### 主要发现

在2D和3D点云上成功发现了离散群，包括通过复域上流匹配实现的反射；识别出目标模式对称排列导致的'最后时刻收敛'挑战，样本在流中保持静止直到相对较晚阶段；提出了解决这一挑战的新插值方案。

### 结论

LieFlow方法能够有效地从数据中学习对称性，在灵活性和假设要求方面优于先前方法，为对称性发现提供了新的途径。

### 翻译

对称性是理解物理系统的基础，同时也能提高机器学习的性能和样本效率。这两种追求都需要了解数据中潜在的对称性。为此，我们提出通过在李群上进行流匹配直接从数据中学习对称性。我们将对称性发现表述为学习更大假设组上的分布，使学习分布匹配数据中观察到的对称性。与先前工作相比，我们的方法LieFlow在可发现的群类型方面更灵活，且需要更少的假设。在2D和3D点云上的实验成功发现了离散群，包括通过复域上的流匹配实现的反射。我们确定了一个关键挑战：目标模式的对称排列导致'最后时刻收敛'，其中样本在流中保持静止直到相对较晚的阶段，并引入了一种用于对称性发现的流匹配的新插值方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何直接从数据中学习对称性的问题，特别是李群对称性。这个问题在物理学和机器学习中都非常重要，因为对称性是理解物理系统的基础，同时也能提高机器学习的性能和样本效率。然而，现有方法通常需要预先知道确切的对称群，而实际应用中潜在的对称性往往是未知的、近似的或特定于领域的，如在材料化学中减少搜索空间，或在计算机视觉中处理有噪声的部分扫描数据。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有对称性发现方法的局限性，如只能在特定设置中工作或需要强假设。他们借鉴了流匹配(flow matching)方法，特别是Lipman等人的工作，将其扩展到李群流形上。作者将对称性发现问题重新表述为在李群上的分布学习问题，通过流匹配将先验分布转换到与数据中观察到的对称性相匹配的分布。虽然借鉴了现有工作，但LieFlow解决了之前方法的局限性，如固定的李代数基础假设、不可解释的对称性和特定分布假设等。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将对称性发现转化为李群上的流匹配问题，学习一个分布在更大的假设群上，该分布集中在实际存在于数据中的子群上。整体实现流程包括：1)训练过程：从数据采样x1，从先验群采样g，计算x0=gx1，采样时间t，计算曲线上的点xt，通过神经网络预测变换并计算损失；2)生成过程：类似训练但生成新样本和累积变换；3)群元素生成：组合变换得到与目标群一致的元素。整个过程直接在李群流形上操作，条件是数据样本。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)首次将对称性发现表述为李群上的流匹配问题；2)提供统一框架发现连续和离散对称性，包括通过复域流匹配发现反射；3)识别并解决'最后时刻模式收敛'问题，引入新时间调度；4)直接在李群流形上学习流。相比之前工作，LieFlow更灵活(不限制群类型)，假设更少(不固定李代数基础)，产生可解释对称性，使用单阶段优化而非两阶段，且在流形而非欧几里得空间操作。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LieFlow通过将对称性发现转化为李群上的流匹配问题，提供了一个统一框架来自动从数据中发现连续和离散对称性，解决了物理学和机器学习中需要预先知道对称性的关键限制。'}


### 论文摘要

Symmetry is fundamental to understanding physical systems, and at the same time, can improve performance and sample efficiency in machine learning. Both pursuits require knowledge of the underlying symmetries in data. To address this, we propose learning symmetries directly from data via flow matching on Lie groups. We formulate symmetry discovery as learning a distribution over a larger hypothesis group, such that the learned distribution matches the symmetries observed in data. Relative to previous works, our method, \lieflow, is more flexible in terms of the types of groups it can discover and requires fewer assumptions. Experiments on 2D and 3D point clouds demonstrate the successful discovery of discrete groups, including reflections by flow matching over the complex domain. We identify a key challenge where the symmetric arrangement of the target modes causes ``last-minute convergence,'' where samples remain stationary until relatively late in the flow, and introduce a novel interpolation scheme for flow matching for symmetry discovery.

---

## 25. LiteGE: Lightweight Geodesic Embedding for Efficient Geodesics Computation and Non-Isometric Shape Correspondence

**论文链接:** [http://arxiv.org/abs/2512.17781v2](http://arxiv.org/abs/2512.17781v2)

**作者:** Yohanes Yudhi Adikusuma, Qixing Huang, Ying He

**发布时间:** 2025-12-19

### GPT解析

### 总结

本文提出了LiteGE，一种轻量级的3D表面测地距离计算方法，通过PCA分析UDF样本构建紧凑形状描述符，显著降低内存使用和计算延迟，同时保持高准确度。

### 背景

计算3D表面测地距离是3D视觉和几何处理的基础任务，与形状对应等密切相关。现有基于学习的方法表现优异但依赖大型3D骨干网络，导致高内存使用和延迟，限制了在交互式或资源受限环境中的应用。

### 目的

开发一种轻量级方法，解决现有测地距离计算方法的高内存占用和高延迟问题，使其适用于资源受限环境。

### 方法

LiteGE通过将主成分分析应用于信息体素处的无符号距离场样本，构建紧凑的、类别感知的形状描述符。这种方法计算效率高，无需高容量网络，且支持稀疏点云输入。

### 主要发现

LiteGE在仅300个点的稀疏点云上仍保持鲁棒性；相比现有神经方法，内存使用和推理时间减少高达300倍；实现了与最先进基于网格方法相比高达1000倍的加速，同时在非等距形状对上保持相当准确性。

### 结论

LiteGE是一种高效、准确的测地距离计算解决方案，特别适合资源受限环境，在点云输入上表现优异，为3D视觉和几何处理任务提供了实用工具。

### 翻译

在3D表面上计算测地距离是3D视觉和几何处理中许多任务的基础，与形状对应等任务有深度联系。最近基于学习的方法表现良好，但依赖于大型3D骨干网络，导致高内存使用和延迟，限制了它们在交互式或资源受限环境中的使用。我们引入了LiteGE，一种轻量级方法，通过将主成分分析应用于信息体素处的无符号距离场样本，构建紧凑的、类别感知的形状描述符。这种描述符计算效率高，且不需要高容量网络。LiteGE在稀疏点云上保持鲁棒性，支持仅300个点的输入，而先前的方法在此情况下会失败。大量实验表明，与现有神经方法相比，LiteGE将内存使用和推理时间减少了高达300倍。此外，通过利用测地距离和形状对应之间的内在关系，LiteGE实现了快速且准确的形状匹配。与最先进的基于网格的方法相比，我们的方法实现了高达1000倍的加速，同时在非等距形状对上保持了相当的准确性，包括在点云输入上的评估。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何高效计算3D表面上的测地距离问题。这个问题很重要，因为测地距离是3D形状的基本属性，广泛应用于形状匹配、表面重建、参数化、纹理映射、分割和3D卷积网络构建等多个领域。现有方法要么计算速度慢，要么内存消耗大，限制了在交互式或资源受限环境中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有基于深度学习方法的局限性（内存消耗高、推理时间长），观察到形状在特定类别中共享相似几何结构，通过形状标准化可减少数据变化。作者借鉴了NeuroGF和GeGNN的嵌入公式，但避免了使用大型3D网络；借鉴了UDF表示方法；在形状匹配中使用了T-Net对齐。创新点在于使用PCA处理UDF表示创建紧凑形状嵌入，而非依赖大型3D网络。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用PCA对无符号距离场(UDF)样本进行降维，创建紧凑的形状描述符，并通过形状标准化减少数据变化。实现流程包括：1)形状标准化（中心化、缩放、方向对齐）；2)选择有信息量体素并计算UDF；3)应用PCA降维；4)使用轻量级MLP预测测地距离；5)对于形状匹配，采用从粗到细的策略和多级最近邻缓存；6)支持测地路径追踪。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)使用PCA处理UDF创建紧凑形状嵌入，减少内存和计算时间；2)在稀疏点云(仅300点)上表现良好；3)直接使用测地距离作为非等距形状匹配的主要监督信号；4)针对不同任务设计形状标准化方法；5)多级最近邻缓存优化形状匹配。相比传统方法，LiteGE计算速度更快；相比现有深度学习方法，内存消耗更低；在形状匹配上，直接使用测地距离作为主要监督信号而非仅作为正则化项，且同时支持网格和点云输入。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LiteGE通过创新的PCA处理UDF表示方法，实现了高效、轻量级的测地距离计算和非等距形状匹配，在保持与现有方法相当准确性的同时，显著降低了内存使用和计算时间。'}


### 论文摘要

Computing geodesic distances on 3D surfaces is fundamental to many tasks in 3D vision and geometry processing, with deep connections to tasks such as shape correspondence. Recent learning-based methods achieve strong performance but rely on large 3D backbones, leading to high memory usage and latency, which limit their use in interactive or resource-constrained settings. We introduce LiteGE, a lightweight approach that constructs compact, category-aware shape descriptors by applying Principal Component Analysis (PCA) to unsigned distance field (UDFs) samples at informative voxels. This descriptor is efficient to compute and removes the need for high-capacity networks. LiteGE remains robust on sparse point clouds, supporting inputs with as few as 300 points, where prior methods fail. Extensive experiments show that LiteGE reduces memory usage and inference time by up to 300$\times$ compared to existing neural approaches. In addition, by exploiting the intrinsic relationship between geodesic distance and shape correspondence, LiteGE enables fast and accurate shape matching. Our method achieves up to 1000$\times$ speedup over state-of-the-art mesh-based approaches while maintaining comparable accuracy on non-isometric shape pairs, including evaluations on point-cloud inputs.

---

## 26. From Theory to Throughput: CUDA-Optimized APML for Large-Batch 3D Learning

**论文链接:** [http://arxiv.org/abs/2512.19743v1](http://arxiv.org/abs/2512.19743v1)

**作者:** Sasan Sharifipour, Constantino Álvarez Casado, Manuel Lage Cañellas, Miguel Bordallo López

**发布时间:** 2025-12-17

**备注:** 5 pages, 2 figures, 2 tables, 5 formulas, 34 references, journal paper

### GPT解析

### 总结

本文提出了一种名为CUDA-APML的稀疏GPU实现，用于近似最优传输点云匹配，显著减少了内存使用同时保持匹配精度。

### 背景

在3D点云模型学习中，损失函数至关重要。常见的Chamfer Distance计算高效但允许多对一对应，而Earth Mover Distance更准确但计算成本高。APML方法使用可微分Sinkhorn迭代和解析温度近似传输，但其密集公式在内存上呈二次方扩展。

### 目的

开发一种内存高效的GPU实现，减少APML方法的内存使用，同时保持其匹配精度和梯度保留能力。

### 方法

CUDA-APML是一种稀疏GPU实现，对可忽略的分配进行阈值处理，直接在COO形式下运行自适应softmax、双向对称化和Sinkhorn归一化，实现近线性内存扩展并保留存储支持上的梯度。

### 主要发现

在ShapeNet和MM-Fi数据集上，CUDA-APML在较小容差范围内匹配密集APML的性能，同时将GPU峰值内存减少了99.9%。点对距离评估目前仍是二次方的。

### 结论

CUDA-APML成功解决了APML方法的内存扩展问题，实现了近线性的内存使用，同时保持了高精度的点云匹配能力。

### 翻译

损失函数是学习准确的3D点云模型的基础，然而常见的选择在几何保真度和计算成本之间进行权衡。Chamfer Distance效率高但允许多对一对应关系，而Earth Mover Distance更好地反映一对一传输但计算成本高。APML使用可微分的Sinkhorn迭代和解析推导的温度近似传输，但其密集公式在内存上呈二次方扩展。我们提出了CUDA-APML，一个稀疏GPU实现，它对可忽略的分配进行阈值处理，并直接在COO形式下运行自适应softmax、双向对称化和Sinkhorn归一化。这实现了近线性的内存扩展并保留了存储支持上的梯度，而点对距离评估在当前实现中仍然是二次方的。在ShapeNet和MM-Fi上，CUDA-APML在较小容差范围内匹配密集APML，同时将GPU峰值内存减少了99.9%。代码可在以下网址获取：https://github.com/Multimodal-Sensing-Lab/apml

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D点云学习中损失函数的内存效率问题。具体来说，APML（自适应概率匹配损失）虽然能提供准确的3D点云学习，但其密集实现方式在内存上呈二次方增长，限制了在大规模点云和大规模批处理训练中的应用。这个问题在现实中很重要，因为3D点云广泛应用于机器人、AR/VR、无线人类感知和数字孪生等领域，而内存限制会限制批处理大小和点数，增加训练成本，阻碍高分辨率场景的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：Chamfer Distance效率高但允许多对一分配，Earth Mover Distance更好地反映一对一匹配但计算成本高。作者注意到APML在自适应softmax阶段后，未归一化的相似度矩阵中的大多数条目接近于零，这是一个经验性质。基于这一观察，作者设计了稀疏实现，通过修剪微不足道的相似度条目来减少内存使用。作者借鉴了APML最优传输理论、稀疏表示和GPU优化技术（如FlashAttention和KeOps）以及稀疏张量引擎等现有工作，将其应用于解决内存效率问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用APML在自适应softmax阶段后的经验稀疏性，通过修剪微不足道的相似度条目来减少内存使用，同时在GPU上直接处理稀疏数据结构，避免创建和存储密集的N×M张量。整体实现流程包括：1)行方向处理，扫描每一行计算最小值和温度，识别并保存重要配对；2)列方向处理，交换角色重复上述步骤；3)对称化，连接COO条目并排序；4)Sinkhorn迭代，在稀疏支持上执行列缩放和行缩放；5)损失和反向传播，在COO配对上评估损失并通过稀疏运算反向传播。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)稀疏GPU实现CUDA-APML，通过修剪微不足道的分配直接在COO格式中执行各种操作；2)实现了接近线性的内存扩展，同时保持梯度，将峰值GPU内存减少99.9%；3)在保持精度的同时显著提高内存效率；4)使基于传输的监督在实际应用中更加可行。相比之前的工作，CUDA-APML不仅保留了APML的准确性，还解决了其内存瓶颈问题，使其能够处理更大的点云和批处理大小，而无需引入额外的近似或设计选择。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文贡献了一种名为CUDA-APML的高效稀疏GPU实现，通过利用最优传输的经验稀疏性，在保持APML精度的同时将内存使用减少了99.9%，使得大规模3D点云学习在实际应用中变得可行。'}


### 论文摘要

Loss functions are fundamental to learning accurate 3D point cloud models, yet common choices trade geometric fidelity for computational cost. Chamfer Distance is efficient but permits many-to-one correspondences, while Earth Mover Distance better reflects one-to-one transport at high computational cost. APML approximates transport with differentiable Sinkhorn iterations and an analytically derived temperature, but its dense formulation scales quadratically in memory. We present CUDA-APML, a sparse GPU implementation that thresholds negligible assignments and runs adaptive softmax, bidirectional symmetrization, and Sinkhorn normalization directly in COO form. This yields near-linear memory scaling and preserves gradients on the stored support, while pairwise distance evaluation remains quadratic in the current implementation. On ShapeNet and MM-Fi, CUDA-APML matches dense APML within a small tolerance while reducing peak GPU memory by 99.9%. Code available at: https://github.com/Multimodal-Sensing-Lab/apml

---

## 27. 论文ID: 2512.20531v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.20531v1.json'

---

## 28. Modeling Bank Systemic Risk of Emerging Markets under Geopolitical Shocks: Empirical Evidence from BRICS Countries

**论文链接:** [http://arxiv.org/abs/2512.20515v1](http://arxiv.org/abs/2512.20515v1)

**作者:** Haibo Wang

**发布时间:** 2025-12-23

**备注:** 22 pages and 7 figures

### GPT解析

### 总结

BRIDGES框架是一种创新的系统性风险分析方法，能够捕捉金砖国家银行系统的复杂动态和潜在风险，基于信息复杂性层次进行分析和评估。

### 背景

金砖国家（BRICS）的经济影响力日益增长，需要能够捕捉复杂、长期动态的风险模型来评估金融系统稳定性。

### 目的

介绍BRIDGES（Bank Risk Interlinkage with Dynamic Graph and Event Simulations）框架，用于分析基于信息复杂性层次的系统性风险。

### 方法

BRIDGES框架使用动态时间规整（DTW）距离构建基于551家BRICS银行战略相似性的动态网络；利用零阶信息（如资产负债表数据）构建网络，一阶信息（如风险比率趋势）检测行为变化，时序图神经网络（TGNN）学习网络演化并检测二阶信息（如结构关系异常），最后通过基于主体的模型（ABM）模拟评估系统韧性。

### 主要发现

最大机构的失败比财务脆弱或动态异常机构失败造成更多系统性损害，由恐慌效应驱动；具有相关国家传播的地缘政治冲击比'太大而不能倒闭'场景造成更严重的系统性损害，近乎导致系统完全崩溃。

### 结论

对BRICS金融稳定的主要威胁是二阶恐慌和大范围地缘政治冲击，这些威胁可能被传统风险分析模型忽视。

### 翻译

金砖国家日益增长的经济影响力需要能够捕捉复杂、长期动态的风险模型。本文介绍了银行风险动态图与事件模拟互联（BRIDGES）框架，该框架基于信息复杂性层次（零阶、一阶和二阶）分析系统性风险。BRIDGES利用动态时间规整（DTW）距离构建551家金砖国家银行的动态网络，基于其战略相似性，使用零阶信息如2008年至2024年的年度资产负债表数据。然后采用一阶信息，包括风险比率趋势，来检测银行行为的变化。作为BRIDGES核心的时序图神经网络（TGNN）被部署以学习网络演化并检测二阶信息，如银行网络结构关系的异常变化。为了衡量异常变化对网络稳定性的影响，BRIDGES进行基于主体的模型（ABM）模拟，评估银行系统对内部金融失败和外部地缘政治冲击在国家个体层面和整个金砖国家范围内的韧性。模拟结果表明，最大机构的失败比财务脆弱或动态异常机构的失败造成更多系统性损害，由强大的恐慌效应驱动。与这种'太大而不能倒闭'的情景相比，具有相关国家级传播的地缘政治冲击会造成更具破坏性的系统性损害，导致近乎完全的系统崩溃。这表明，对金砖国家金融稳定的主要威胁是二阶恐慌和大范围地缘政治冲击，传统风险分析模型可能无法检测到这些威胁。


### 论文摘要

The growing economic influence of the BRICS nations requires risk models that capture complex, long-term dynamics. This paper introduces the Bank Risk Interlinkage with Dynamic Graph and Event Simulations (BRIDGES) framework, which analyzes systemic risk based on the level of information complexity (zero-order, first-order, and second-order). BRIDGES utilizes the Dynamic Time Warping (DTW) distance to construct a dynamic network for 551 BRICS banks based on their strategic similarity, using zero-order information such as annual balance sheet data from 2008 to 2024. It then employs first-order information, including trends in risk ratios, to detect shifts in banks' behavior. A Temporal Graph Neural Network (TGNN), as the core of BRIDGES, is deployed to learn network evolutions and detect second-order information, such as anomalous changes in the structural relationships of the bank network. To measure the impact of anomalous changes on network stability, BRIDGES performs Agent-Based Model (ABM) simulations to assess the banking system's resilience to internal financial failure and external geopolitical shocks at the individual country level and across BRICS nations. Simulation results show that the failure of the largest institutions causes more systemic damage than the failure of the financially vulnerable or dynamically anomalous ones, driven by powerful panic effects. Compared to this "too big to fail" scenario, a geopolitical shock with correlated country-wide propagation causes more destructive systemic damage, leading to a near-total systemic collapse. It suggests that the primary threats to BRICS financial stability are second-order panic and large-scale geopolitical shocks, which traditional risk analysis models might not detect.

---

## 29. A Novel Graph-Sequence Learning Model for Inductive Text Classification

**论文链接:** [http://arxiv.org/abs/2512.20097v1](http://arxiv.org/abs/2512.20097v1)

**作者:** Zuo Wang, Ye Yuan

**发布时间:** 2025-12-23

### GPT解析

### 总结

本文提出了一种新颖的图-序列学习模型(TextGSL)用于归纳文本分类，通过结合图神经网络和Transformer的优势，解决了现有方法在处理多样化结构信息和序列信息方面的局限性，提高了文本分类的准确性。

### 背景

文本分类在情感分析、假新闻检测和公共舆论分析等多种下游任务中扮演重要角色。基于图神经网络(GNNs)的文本分类最近取得了显著进展，因其具有强大的结构关系学习能力。

### 目的

解决现有基于GNN的文本分类方法的两大局限性：未能充分考虑单词对之间的多样化结构信息，以及在文本图结构信息学习模块中忽略序列信息的问题。

### 方法

为每个文本中的所有单词构建单一文本级图，基于单词对之间的不同关系建立不同的边类型；设计自适应多边消息传递范式聚合多样化结构信息；通过融入Transformer层捕获文本数据中的序列信息。

### 主要发现

TextGSL能够学习更具区分性的文本表示；在多样化的基准数据集上与多个强基线进行比较，实验结果表明TextGSL在准确性方面优于这些基线。

### 结论

TextGSL是一种有效的文本分类方法，通过结合图神经网络和Transformer的优势，解决了现有方法的局限性，提高了分类性能，尤其擅长处理包含新单词和新关系的文本分类任务。

### 翻译

文本分类在各种下游文本相关任务中扮演着重要角色，例如情感分析、假新闻检测和公共舆论分析。最近，基于图神经网络(GNNs)的文本分类由于其强大的结构关系学习能力取得了显著进展。然而，这些方法仍面临两个主要局限。首先，这些方法未能充分考虑单词对之间的多样化结构信息，例如共现、句法和语义。此外，它们在文本图结构信息学习模块中忽略了序列信息，无法对新单词和新关系进行文本分类。在本文中，我们提出了一种新颖的图-序列学习模型用于归纳文本分类(TextGSL)来解决上述问题。更具体地说，我们为每个文本中的所有单词构建单一文本级图，并基于单词对之间的不同关系建立不同的边类型。在此基础上，我们设计了一个自适应多边消息传递范式来聚合单词对之间的多样化结构信息。此外，通过融入Transformer层，所提出的TextGSL可以捕获文本数据中的序列信息。因此，TextGSL能够学习更具区分性的文本表示。TextGSL已与多个强基线进行了全面比较。在多样化基准数据集上的实验结果表明，TextGSL在准确性方面优于这些基线。


### 论文摘要

Text classification plays an important role in various downstream text-related tasks, such as sentiment analysis, fake news detection, and public opinion analysis. Recently, text classification based on Graph Neural Networks (GNNs) has made significant progress due to their strong capabilities of structural relationship learning. However, these approaches still face two major limitations. First, these approaches fail to fully consider the diverse structural information across word pairs, e.g., co-occurrence, syntax, and semantics. Furthermore, they neglect sequence information in the text graph structure information learning module and can not classify texts with new words and relations. In this paper, we propose a Novel Graph-Sequence Learning Model for Inductive Text Classification (TextGSL) to address the previously mentioned issues. More specifically, we construct a single text-level graph for all words in each text and establish different edge types based on the diverse relationships between word pairs. Building upon this, we design an adaptive multi-edge message-passing paradigm to aggregate diverse structural information between word pairs. Additionally, sequential information among text data can be captured by the proposed TextGSL through the incorporation of Transformer layers. Therefore, TextGSL can learn more discriminative text representations. TextGSL has been comprehensively compared with several strong baselines. The experimental results on diverse benchmarking datasets demonstrate that TextGSL outperforms these baselines in terms of accuracy.

---

## 30. Jensen-Shannon Divergence Message-Passing for Rich-Text Graph Representation Learning

**论文链接:** [http://arxiv.org/abs/2512.20094v1](http://arxiv.org/abs/2512.20094v1)

**作者:** Zuo Wang, Ye Yuan

**发布时间:** 2025-12-23

### GPT解析

### 总结

本文提出了一种名为Jensen-Shannon散度消息传递(JSDMP)的新学习范式，用于处理富文本图中的表示学习问题，通过同时考虑相似性和不相似性来改进节点表示

### 背景

广泛存在的上下文和结构差异可能影响富文本图中的表示学习效果

### 目的

开发一种能够有效捕捉上下文和结构差异的学习范式，提升富文本图表示学习的质量

### 方法

提出JSDMP学习范式，结合Jensen-Shannon散度捕获结构和文本的相似性与不相似性，并基于此开发了两种图神经网络：DMPGCN和DMPPRG

### 主要发现

在富文本数据集上的实验表明，DMPGCN和DMPPRG性能优于多种最先进的基线方法

### 结论

Jensen-Shannon散度消息传递范式能有效提升富文本图表示学习的效果

### 翻译

在本文中，我们研究了广泛存在的上下文和结构差异如何影响富文本图中的表示学习。为此，我们提出了Jensen-Shannon散度消息传递(JSDMP)，一种用于富文本图表示学习的新学习范式。除了考虑结构和文本的相似性外，JSDMP还通过Jensen-Shannon散度进一步捕获它们对应的不相似性。然后，相似性和不相似性被联合用于计算文本节点之间的新消息权重，从而使表示能够从真正相关的文本节点学习上下文和结构信息。基于JSDMP，我们提出了两种新颖的图神经网络，即差异消息传递图卷积网络(DMPGCN)和差异消息传递PageRank图神经网络(DMPPRG)，用于学习富文本图中的表示。DMPGCN和DMPPRG已在成熟的富文本数据集上进行了广泛测试，并与几种最先进的基线方法进行了比较。实验结果表明，DMPGCN和DMPPRG能够优于其他基线方法，证明了所提出的Jensen-Shannon散度消息传递范式的有效性


### 论文摘要

In this paper, we investigate how the widely existing contextual and structural divergence may influence the representation learning in rich-text graphs. To this end, we propose Jensen-Shannon Divergence Message-Passing (JSDMP), a new learning paradigm for rich-text graph representation learning. Besides considering similarity regarding structure and text, JSDMP further captures their corresponding dissimilarity by Jensen-Shannon divergence. Similarity and dissimilarity are then jointly used to compute new message weights among text nodes, thus enabling representations to learn with contextual and structural information from truly correlated text nodes. With JSDMP, we propose two novel graph neural networks, namely Divergent message-passing graph convolutional network (DMPGCN) and Divergent message-passing Page-Rank graph neural networks (DMPPRG), for learning representations in rich-text graphs. DMPGCN and DMPPRG have been extensively texted on well-established rich-text datasets and compared with several state-of-the-art baselines. The experimental results show that DMPGCN and DMPPRG can outperform other baselines, demonstrating the effectiveness of the proposed Jensen-Shannon Divergence Message-Passing paradigm

---

## 31. 论文ID: 2512.20086v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.20086v1.json'

---

## 32. QE-Catalytic: A Graph-Language Multimodal Base Model for Relaxed-Energy Prediction in Catalytic Adsorption

**论文链接:** [http://arxiv.org/abs/2512.20084v1](http://arxiv.org/abs/2512.20084v1)

**作者:** Yanjie Li, Jian Xu, Xueqing Chen, Lina Yu, Shiming Xiang, Weijun Li, Cheng-lin Liu

**发布时间:** 2025-12-23

**备注:** 25 pages

### GPT解析

### 总结

QE-Catalytic是一个多模态框架，深度耦合大型语言模型Qwen与E(3)等变图Transformer Equiformer-V2，用于复杂催化表面的吸附构型属性预测和逆向设计。

### 背景

吸附能是催化反应性的关键描述符，其准确性直接影响机器学习驱动的催化剂筛选可靠性。E(3)等变图神经网络在三维原子坐标操作上表现良好，而基于语言模型的方法在吸附构型能量预测和区分相同系统不同构型方面存在不足。

### 目的

开发一个能够同时支持吸附构型属性预测和逆向设计的多模态框架，提高吸附能预测的准确性。

### 方法

QE-Catalytic联合利用三维结构和结构化配置文本，通过图-文本对齐将3D几何信息注入语言通道，使模型在精确坐标不可用时仍可作为高性能基于文本的预测器，并能自回归生成目标能量驱动的结构设计和信息补全的CIF文件。

### 主要发现

在OC20数据集上，QE-Catalytic将松弛吸附能的平均绝对误差从0.713 eV降低到0.486 eV，并在多种评估协议中持续优于CatBERTa和GAP-CATBERTa等基线模型。

### 结论

QE-Catalytic有效结合了语言模型和图神经网络的优势，在吸附能预测和逆向设计方面表现优异，为催化剂筛选提供了更可靠的工具。

### 翻译

吸附能是催化反应性的关键描述符。它从根本上定义为吸附物-表面系统的松弛总能量与适当参考状态之间的差异；因此，松弛能量预测的准确性直接决定了机器学习驱动的催化剂筛选的可靠性。E(3)等变图神经网络可以在周期性边界条件下原生操作三维原子坐标，并在此类任务上表现出强大的性能。相比之下，基于语言模型的方法虽然能够生成人类可读的文本描述并减少对显式图的依赖——从而扩大了适用范围——但在吸附构型能量预测准确性和区分'相同系统不同构型'方面仍然不足，即使在GAP-CATBERTa风格的图辅助预训练下也是如此。为此，我们提出了QE-Catalytic，一个多模态框架，深度耦合大型语言模型(Qwen)与E(3)等变图Transformer(Equiformer-V2)，能够统一支持复杂催化表面上吸附构型属性预测和逆向设计。在预测过程中，QE-Catalytic联合利用三维结构和结构化配置文本，并通过图-文本对齐将'3D几何信息'注入语言通道，使其在精确坐标不可用时可作为高性能基于文本的预测器，同时也能自回归生成目标能量驱动的结构设计和信息补全的CIF文件。在OC20上，QE-Catalytic将松弛吸附能的MAE从0.713 eV降低到0.486 eV，并且在多种评估协议中持续优于CatBERTa和GAP-CATBERTa等基线模型。


### 论文摘要

Adsorption energy is a key descriptor of catalytic reactivity. It is fundamentally defined as the difference between the relaxed total energy of the adsorbate-surface system and that of an appropriate reference state; therefore, the accuracy of relaxed-energy prediction directly determines the reliability of machine-learning-driven catalyst screening. E(3)-equivariant graph neural networks (GNNs) can natively operate on three-dimensional atomic coordinates under periodic boundary conditions and have demonstrated strong performance on such tasks. In contrast, language-model-based approaches, while enabling human-readable textual descriptions and reducing reliance on explicit graph -- thereby broadening applicability -- remain insufficient in both adsorption-configuration energy prediction accuracy and in distinguishing ``the same system with different configurations,'' even with graph-assisted pretraining in the style of GAP-CATBERTa.   To this end, we propose QE-Catalytic, a multimodal framework that deeply couples a large language model (\textbf{Q}wen) with an E(3)-equivariant graph Transformer (\textbf{E}quiformer-V2), enabling unified support for adsorption-configuration property prediction and inverse design on complex catalytic surfaces. During prediction, QE-Catalytic jointly leverages three-dimensional structures and structured configuration text, and injects ``3D geometric information'' into the language channel via graph-text alignment, allowing it to function as a high-performance text-based predictor when precise coordinates are unavailable, while also autoregressively generating CIF files for target-energy-driven structure design and information completion. On OC20, QE-Catalytic reduces the MAE of relaxed adsorption energy from 0.713~eV to 0.486~eV, and consistently outperforms baseline models such as CatBERTa and GAP-CATBERTa across multiple evaluation protocols.

---

## 33. MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis

**论文链接:** [http://arxiv.org/abs/2512.20026v1](http://arxiv.org/abs/2512.20026v1)

**作者:** Ziwei Qin, Xuhui Song, Deqing Huang, Na Qin, Jun Li

**发布时间:** 2025-12-23

**备注:** Accepted by Proceedings of the AAAI Conference on Artificial Intelligence 40 (AAAI-26)

### GPT解析

### 总结

本文提出了一种多激活平面交互图神经网络(MAPI-GNN)，通过学习多方面图配置来改进多模态医疗诊断中的图神经网络应用，解决了传统方法依赖单一静态图的局限性。

### 背景

图神经网络因其固有关系建模能力越来越多地应用于多模态医疗诊断，但其功效常因依赖从无差别特征构建的单个静态图而受限，无法有效建模患者特定病理关系。

### 目的

重建传统的单图范式，通过从语义解耦的特征子空间学习多方面图配置，提高医疗诊断的准确性和患者特异性。

### 方法

MAPI-GNN框架首先通过多维鉴别器发现潜在图感知模式，这些模式指导动态构建激活图堆栈，最后通过关系融合引擎对多方面配置进行聚合和上下文化，实现稳健诊断。

### 主要发现

在两个包含超过1300个患者样本的不同任务上的大量实验表明，MAPI-GNN显著优于现有最先进方法。

### 结论

通过学习多方面图配置而非依赖单一静态图，MAPI-GNN有效解决了传统图神经网络在多模态医疗诊断中的局限性，提高了诊断性能。

### 翻译

图神经网络因其固有的关系建模能力，越来越多地被应用于多模态医疗诊断。然而，它们的有效性常常因为普遍依赖从无差别特征构建的单个静态图而受到损害，这阻碍了建模患者特定病理关系的能力。为此，提出的多激活平面交互图神经网络(MAPI-GNN)通过从语义解耦的特征子空间学习多方面图配置来重建这种单图范式。该框架首先通过多维鉴别器发现潜在的图感知模式；这些模式然后指导动态构建一堆激活图；最后，这种多方面配置通过关系融合引擎进行聚合和上下文化，以实现稳健诊断。在两个包含超过1300个患者样本的不同任务上的大量实验表明，MAPI-GNN显著优于最先进的方法。


### 论文摘要

Graph neural networks are increasingly applied to multimodal medical diagnosis for their inherent relational modeling capabilities. However, their efficacy is often compromised by the prevailing reliance on a single, static graph built from indiscriminate features, hindering the ability to model patient-specific pathological relationships. To this end, the proposed Multi-Activation Plane Interaction Graph Neural Network (MAPI-GNN) reconstructs this single-graph paradigm by learning a multifaceted graph profile from semantically disentangled feature subspaces. The framework first uncovers latent graph-aware patterns via a multi-dimensional discriminator; these patterns then guide the dynamic construction of a stack of activation graphs; and this multifaceted profile is finally aggregated and contextualized by a relational fusion engine for a robust diagnosis. Extensive experiments on two diverse tasks, comprising over 1300 patient samples, demonstrate that MAPI-GNN significantly outperforms state-of-the-art methods.

---

## 34. A hybrid global local computational framework for ship hull structural analysis using homogenized model and graph neural network

**论文链接:** [http://arxiv.org/abs/2512.20020v1](http://arxiv.org/abs/2512.20020v1)

**作者:** Yuecheng Cai, Jasmin Jelovica

**发布时间:** 2025-12-23

### GPT解析

### 总结

该研究提出了一种结合等效单层模型和图神经网络的计算框架，用于船体梁的全局-局部结构分析。该框架通过粗网格ESL模型预测全局位移，然后使用异构图变换器预测局部应力场，显著提高了局部分析的效率和准确性。

### 背景

船体梁结构需要进行全局和局部分析来评估其性能，但传统方法在计算效率和精度之间存在权衡。

### 目的

开发一个能够高效且准确地预测船体梁的全局位移场和局部应力、位移场的计算框架，特别适用于优化设计。

### 方法

集成等效单层模型和图神经网络；使用粗网格均质化ESL模型预测全局位移场；开发全局到局部自由度映射和重建程序；使用异构图变换器作为局部分析工具；使用高保真3D面板有限元模型训练HGT；仅需全局ESL解即可生成详细局部响应。

### 主要发现

在三个箱梁案例研究中验证，全局预测误差由粗网格ESL解决定，而HGT保持了高局部精度，明显优于传统的基于ESL的应力估计方法。

### 结论

该框架结合了ESL模型的全局效率和HGT的局部精度，为船体梁结构分析提供了高效且准确的解决方案，特别适用于优化设计。

### 翻译

这项研究提出了一个用于船体梁全局-局部结构分析的计算框架，该框架集成了等效单层模型与图神经网络。粗网格均质化ESL模型可有效预测全局位移场，从中提取加劲板边界上的自由度。开发了全局到局部自由度映射和重建程序，以恢复局部分析的详细边界运动学。重建的自由度与面板几何和载荷一起作为异构图变换器的输入，该变换器能够快速准确地预测船体梁内任何面板的详细应力和位移场。HGT使用具有重建边界条件的高保真3D面板有限元模型进行训练，使其能够泛化到不同的面板几何形状、载荷和边界行为。一旦训练完成，该框架仅需全局ESL解即可生成详细的局部响应，使其非常适用于优化。


### 论文摘要

This study presents a computational framework for global local structural analysis of ship hull girders that integrates an equivalent single layer (ESL) model with a graph neural network (GNN). A coarse mesh homogenized ESL model efficiently predicts the global displacement field, from which degrees of freedom (DOFs) along stiffened panel boundaries are extracted. A global to local DOF mapping and reconstruction procedure is developed to recover detailed boundary kinematics for local analysis. The reconstructed DOFs, together with panel geometry and loading, serve as inputs to a heterogeneous graph transformer (HGT), a subtype of GNN, which rapidly and accurately predicts the detailed stress and displacement fields for any panel within the hull girder. The HGT is trained using high fidelity 3D panel finite element model with reconstructed boundary conditions, enabling it to generalize across varying panel geometries, loadings, and boundary behaviors. Once trained, the framework requires only the global ESL solution in order to generate detailed local responses, making it highly suitable for optimization. Validation on three box beam case studies demonstrates that the global prediction error is governed by the coarse mesh ESL solution, while the HGT maintains high local accuracy and clearly outperforms conventional ESL based stress estimation method.

---

## 35. 论文ID: 2512.20004v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.20004v1.json'

---

## 36. Spatio-Temporal Graph Neural Networks for Dairy Farm Sustainability Forecasting and Counterfactual Policy Analysis

**论文链接:** [http://arxiv.org/abs/2512.19970v1](http://arxiv.org/abs/2512.19970v1)

**作者:** Surya Jayakumar, Kieran Sullivan, John McLaughlin, Christine O'Meara, Indrakshi Dey

**发布时间:** 2025-12-23

### GPT解析

### 总结

本研究引入了一种新颖的数据驱动框架，首次在县级规模应用时空图神经网络(STGNN)预测综合可持续性指数，基于畜群级运营记录。

### 背景

畜牧业可持续发展评估需要更精确和全面的预测方法，传统方法难以处理复杂的时空依赖关系和数据稀疏性问题。

### 目的

开发一种能够从畜群级运营记录中预测综合可持续性指数的框架，为爱尔兰牛育种联合会提供2026-2030年的多年预测。

### 方法

采用变分自编码器(VAE)增强数据集，通过主成分分析确定四个关键评分维度(繁殖效率、遗传管理、畜群健康和畜群管理)，构建加权综合指数，并使用新型STGNN架构建模时空依赖关系。

### 主要发现

成功开发并应用了首个县级规模的STGNN框架用于可持续性预测，该框架能够有效处理地理依赖性和非线性时间动态，生成可靠的多年预测结果。

### 结论

该数据驱动框架为畜牧业可持续发展评估提供了新方法，能够整合多维度数据并进行长期预测，有助于畜牧业决策和规划。

### 翻译

本研究引入了一种新颖的数据驱动框架和首次在县级规模应用的时空图神经网络(STGNN)，用于从畜群级运营记录预测综合可持续性指数。该方法采用了一种新颖的端到端管道，使用变分自编码器(VAE)增强爱尔兰牛育种联合会(ICBF)数据集，同时保持联合分布并缓解稀疏性。通过主成分分析首次推导出基于支柱的评分公式，确定了繁殖效率、遗传管理、畜群健康和畜群管理，以构建加权综合指数。这些指数使用一种新型STGNN架构进行建模，该架构明确编码了地理依赖性和非线性时间动态，以生成2026-2030年的多年预测。


### 论文摘要

This study introduces a novel data-driven framework and the first-ever county-scale application of Spatio-Temporal Graph Neural Networks (STGNN) to forecast composite sustainability indices from herd-level operational records. The methodology employs a novel, end-to-end pipeline utilizing a Variational Autoencoder (VAE) to augment Irish Cattle Breeding Federation (ICBF) datasets, preserving joint distributions while mitigating sparsity. A first-ever pillar-based scoring formulation is derived via Principal Component Analysis, identifying Reproductive Efficiency, Genetic Management, Herd Health, and Herd Management, to construct weighted composite indices. These indices are modelled using a novel STGNN architecture that explicitly encodes geographic dependencies and non-linear temporal dynamics to generate multi-year forecasts for 2026-2030.

---

## 37. 论文ID: 2512.20308v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.20308v1.json'

---

## 38. 论文ID: 2512.20220v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.20220v1.json'

---

## 39. QuarkAudio Technical Report

**论文链接:** [http://arxiv.org/abs/2512.20151v1](http://arxiv.org/abs/2512.20151v1)

**作者:** Chengwei Liu, Haoyin Yan, Shaofei Xue, Xiaotao Liang, Xiaofu Chen, Bin Gong, Zheng Xue, Gang Song

**发布时间:** 2025-12-23

### GPT解析

### 总结

本文介绍了QuarkAudio，一个基于仅解码器自回归语言模型的统一音频处理和生成框架，能够处理多种音频任务并提供高质量的音频生成能力。

### 背景

现有音频处理和生成模型大多依赖于特定任务架构，导致开发碎片化和扩展性有限。

### 目的

设计一个能处理多任务的统一框架，提供强大的指令和音频理解能力，实现高质量的音频生成。

### 方法

提出QuarkAudio框架，包含统一的离散音频标记器H-Codec，将自监督学习表示整合到标记化和重建过程中；对H-Codec进行改进，包括动态帧率机制和48kHz音频采样率支持；使用特定任务条件信息作为LM条件序列，以自回归方式预测离散音频标记。

### 主要发现

H-Codec以低帧率实现高质量音频重建，提高下游音频生成效率和性能；QuarkAudio在多种任务上达到与最先进系统相当的性能。

### 结论

QuarkAudio是一个有效的统一多任务框架，支持语音修复、目标说话人提取、语音分离、语音转换和语言查询音频源分离等多种任务，并可扩展到自然语言引导的通用音频编辑。

### 翻译

许多现有的音频处理和生成模型依赖于特定任务架构，导致碎片化的开发努力和有限的扩展性。因此，设计一个能够处理多任务、提供强大的指令和音频理解能力以及高质量音频生成的统一框架是很有前景的。这需要兼容的范式设计、强大的骨干网络和高保真音频重建模块。为满足这些需求，本技术报告介绍了QuarkAudio，一个基于仅解码器自回归(AR)语言模型的生成框架，统一了多种任务。该框架包含一个统一的离散音频标记器H-Codec，将自监督学习表示整合到标记化和重建过程中。我们进一步提出了对H-Codec的几项改进，如动态帧率机制和将音频采样率扩展到48kHz。QuarkAudio通过使用特定任务的条件信息作为仅解码器LM的条件序列，并以自回归方式预测离散目标音频标记来统一任务。该框架支持广泛的音频处理和生成任务，包括语音修复(SR)、目标说话人提取(TSE)、语音分离(SS)、语音转换(VC)和语言查询音频源分离(LASS)。此外，我们将下游任务扩展到由自然语言指令引导的通用自由形式音频编辑（包括语音语义编辑和音频事件编辑）。实验结果表明，H-Codec以低帧率实现高质量音频重建，提高了下游音频生成的效率和性能，并且QuarkAudio在多个任务上提供了与最先进的任务特定或多任务系统具有竞争力或可比的性能。


### 论文摘要

Many existing audio processing and generation models rely on task-specific architectures, resulting in fragmented development efforts and limited extensibility. It is therefore promising to design a unified framework capable of handling multiple tasks, while providing robust instruction and audio understanding and high-quality audio generation. This requires a compatible paradigm design, a powerful backbone, and a high-fidelity audio reconstruction module. To meet these requirements, this technical report introduces QuarkAudio, a decoder-only autoregressive (AR) LM-based generative framework that unifies multiple tasks. The framework includes a unified discrete audio tokenizer, H-Codec, which incorporates self-supervised learning (SSL) representations into the tokenization and reconstruction process. We further propose several improvements to H-Codec, such as a dynamic frame-rate mechanism and extending the audio sampling rate to 48 kHz. QuarkAudio unifies tasks by using task-specific conditional information as the conditioning sequence of the decoder-only LM, and predicting discrete target audio tokens in an AR manner. The framework supports a wide range of audio processing and generation tasks, including speech restoration (SR), target speaker extraction (TSE), speech separation (SS), voice conversion (VC), and language-queried audio source separation (LASS). In addition, we extend downstream tasks to universal free-form audio editing guided by natural language instructions (including speech semantic editing and audio event editing). Experimental results show that H-Codec achieves high-quality audio reconstruction with a low frame rate, improving both the efficiency and performance of downstream audio generation, and that QuarkAudio delivers competitive or comparable performance to state-of-the-art task-specific or multi-task systems across multiple tasks.

---

## 40. 论文ID: 2512.20117v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.20117v1.json'

---

## 41. Evolutionary Neural Architecture Search with Dual Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2512.20112v1](http://arxiv.org/abs/2512.20112v1)

**作者:** Xian-Rong Zhang, Yue-Jiao Gong, Wei-Neng Chen, Jun Zhang

**发布时间:** 2025-12-23

**备注:** 26 pages

### GPT解析

### 总结

本文提出了一种名为DCL-ENAS的新方法，通过双阶段对比学习训练神经网络预测器，解决了ENAS中训练数据计算成本高的问题，在多个基准测试和实际任务中取得了优于现有方法的性能。

### 背景

进化神经网络架构搜索(ENAS)因能自动设计神经网络架构而受到关注。现有研究使用神经网络预测器指导搜索过程，但收集训练数据的计算成本很高，因为每个标签都需要完整训练一个架构。在有限计算预算下实现高精度预测器对ENAS成功至关重要。

### 目的

开发一种高效方法，在有限计算预算下训练高精度神经网络预测器，以提升ENAS的性能并降低计算成本。

### 方法

提出ENAS与双对比学习(DCL-ENAS)方法，包含两个阶段：第一阶段使用对比自监督学习从神经网络架构中学习有意义的表示，无需标签；第二阶段使用对比学习进行微调，准确预测不同架构的相对性能而非绝对性能，这足以指导进化搜索。

### 主要发现

在NASBench-101和NASBench-201上，DCL-ENAS实现了最高验证准确率，超越最强已发表基线0.05%(ImageNet16-120)到0.39%(NASBench-101)。在真实ECG心律失常分类任务中，比随机搜索获得的非NAS手动设计模型提高约2.5个百分点，仅需7.7 GPU天。

### 结论

DCL-ENAS通过双阶段对比学习有效解决了ENAS中训练数据计算成本高的问题，在有限计算预算下实现了高性能，为神经网络架构搜索提供了新思路。

### 翻译

进化神经网络架构搜索(ENAS)因能自动设计神经网络架构而受到关注。最近的研究使用神经网络预测器来指导这个过程，但收集训练数据的计算成本很高——因为每个标签都需要完整训练一个架构——这使得在有限计算预算(即完整训练的架构-标签对数量有限)下实现高精度预测器对ENAS成功至关重要。本文引入了带双对比学习的ENAS(DCL-ENAS)，一种使用两个阶段的对比学习来训练神经网络预测器的新方法。在第一阶段，使用对比自监督学习从神经网络架构中学习有意义的表示，不需要标签。在第二阶段，进行对比学习的微调，以准确预测不同架构的相对性能而非绝对性能，这足以指导进化搜索。在NASBench-101和NASBench-201上，DCL-ENAS实现了最高的验证准确率，比最强已发表基线高出0.05%(ImageNet16-120)到0.39%(NASBench-101)。在真实世界的ECG心律失常分类任务中，DCL-ENAS比通过随机搜索获得的非NAS手动设计模型提高了约2.5个百分点的性能，仅需7.7 GPU天。


### 论文摘要

Evolutionary Neural Architecture Search (ENAS) has gained attention for automatically designing neural network architectures. Recent studies use a neural predictor to guide the process, but the high computational costs of gathering training data -- since each label requires fully training an architecture -- make achieving a high-precision predictor with { limited compute budget (i.e., a capped number of fully trained architecture-label pairs)} crucial for ENAS success. This paper introduces ENAS with Dual Contrastive Learning (DCL-ENAS), a novel method that employs two stages of contrastive learning to train the neural predictor. In the first stage, contrastive self-supervised learning is used to learn meaningful representations from neural architectures without requiring labels. In the second stage, fine-tuning with contrastive learning is performed to accurately predict the relative performance of different architectures rather than their absolute performance, which is sufficient to guide the evolutionary search. Across NASBench-101 and NASBench-201, DCL-ENAS achieves the highest validation accuracy, surpassing the strongest published baselines by 0.05\% (ImageNet16-120) to 0.39\% (NASBench-101). On a real-world ECG arrhythmia classification task, DCL-ENAS improves performance by approximately 2.5 percentage points over a manually designed, non-NAS model obtained via random search, while requiring only 7.7 GPU-days.

---

## 42. Vehicle-centric Perception via Multimodal Structured Pre-training

**论文链接:** [http://arxiv.org/abs/2512.19934v1](http://arxiv.org/abs/2512.19934v1)

**作者:** Wentao Wu, Xiao Wang, Chenglong Li, Jin Tang, Bin Luo

**发布时间:** 2025-12-22

**备注:** Journal extension of VehicleMAE (AAAI 2024)

### GPT解析

### 总结

这篇论文提出了VehicleMAE-V2，一种新型的车辆中心预训练大模型，通过利用车辆相关的多模态结构先验来指导掩码令牌重建过程，增强模型学习车辆中心感知的通用表示能力。论文还提出了三个专门设计的模块(SMM、CRM和SRM)来整合对称、轮廓和语义三种结构先验，并构建了一个大规模数据集Autobot4M来支持预训练。实验证明该模型在五个下游任务上表现优越。

### 背景

车辆中心感知在许多智能系统中起着关键作用，包括大规模监控系统、智能交通和自动驾驶。然而，现有方法在预训练过程中缺乏对车辆相关知识的有效学习，导致建模通用车辆感知表示的能力较差。

### 目的

为了解决现有方法在预训练过程中缺乏有效学习车辆相关知识的问题，作者提出了一种名为VehicleMAE-V2的新型车辆中心预训练大模型，旨在增强模型学习车辆中心感知的通用表示能力。

### 方法

作者提出了VehicleMAE-V2模型，并设计了三个专门模块：对称引导掩码模块(SMM)利用车辆对称约束选择高质量掩码图像补丁并减少信息冗余；轮廓引导表示模块(CRM)最小化轮廓特征和重建特征之间的概率分布差异，保持整体车辆结构信息；语义引导表示模块(SRM)通过对比学习和跨模态蒸馏对齐图像文本特征，解决语义理解不足导致的特征混淆问题。此外，作者还构建了Autobot4M数据集，包含约400万车辆图像和12,693个文本描述，以支持预训练。

### 主要发现

在五个下游任务上的大量实验表明，VehicleMAE-V2模型具有优越的性能。通过利用车辆相关的多模态结构先验指导掩码令牌重建过程，模型能够学习到更具通用性的车辆中心感知表示。

### 结论

VehicleMAE-V2通过探索和利用车辆相关的多模态结构先验来指导掩码令牌重建过程，显著增强了模型学习车辆中心感知的通用表示能力。提出的三个模块(SMM、CRM和SRM)分别整合了对称、轮廓和语义三种结构先验，有效解决了预训练过程中的关键问题。Autobot4M数据集的构建也为车辆感知研究提供了宝贵的资源。

### 翻译

车辆中心感知在许多智能系统中起着关键作用，包括大规模监控系统、智能交通和自动驾驶。现有方法在预训练过程中缺乏对车辆相关知识的有效学习，导致建模通用车辆感知表示的能力较差。为了解决这个问题，我们提出了VehicleMAE-V2，一种新型的车辆中心预训练大模型。通过探索和利用车辆相关的多模态结构先验来指导掩码令牌重建过程，我们的方法可以显著增强模型学习车辆中心感知的通用表示能力。具体来说，我们设计了对称引导掩码模块(SMM)、轮廓引导表示模块(CRM)和语义引导表示模块(SRM)，将对称、轮廓和语义三种结构先验分别整合到令牌重建中。SMM利用车辆对称约束避免保留对称补丁，从而可以选择高质量的掩码图像补丁并减少信息冗余。CRM最小化轮廓特征和重建特征之间的概率分布差异，从而可以在像素级重建过程中保持整体车辆结构信息。SRM通过对比学习和跨模态蒸馏对齐图像文本特征，解决掩码重建过程中因语义理解不足导致的特征混淆问题。为了支持VehicleMAE-V2的预训练，我们构建了Autobot4M，一个包含约400万车辆图像和12,693个文本描述的大规模数据集。在五个下游任务上的大量实验证明了VehicleMAE-V2的优越性能。


### 论文摘要

Vehicle-centric perception plays a crucial role in many intelligent systems, including large-scale surveillance systems, intelligent transportation, and autonomous driving. Existing approaches lack effective learning of vehicle-related knowledge during pre-training, resulting in poor capability for modeling general vehicle perception representations. To handle this problem, we propose VehicleMAE-V2, a novel vehicle-centric pre-trained large model. By exploring and exploiting vehicle-related multimodal structured priors to guide the masked token reconstruction process, our approach can significantly enhance the model's capability to learn generalizable representations for vehicle-centric perception. Specifically, we design the Symmetry-guided Mask Module (SMM), Contour-guided Representation Module (CRM) and Semantics-guided Representation Module (SRM) to incorporate three kinds of structured priors into token reconstruction including symmetry, contour and semantics of vehicles respectively. SMM utilizes the vehicle symmetry constraints to avoid retaining symmetric patches and can thus select high-quality masked image patches and reduce information redundancy. CRM minimizes the probability distribution divergence between contour features and reconstructed features and can thus preserve holistic vehicle structure information during pixel-level reconstruction. SRM aligns image-text features through contrastive learning and cross-modal distillation to address the feature confusion caused by insufficient semantic understanding during masked reconstruction. To support the pre-training of VehicleMAE-V2, we construct Autobot4M, a large-scale dataset comprising approximately 4 million vehicle images and 12,693 text descriptions. Extensive experiments on five downstream tasks demonstrate the superior performance of VehicleMAE-V2.

---

## 43. Next-Embedding Prediction Makes Strong Vision Learners

**论文链接:** [http://arxiv.org/abs/2512.16922v2](http://arxiv.org/abs/2512.16922v2)

**作者:** Sihan Xu, Ziqiao Ma, Wenhao Chai, Xuweiyi Chen, Weiyang Jin, Joyce Chai, Saining Xie, Stella X. Yu

**发布时间:** 2025-12-18

**备注:** Project Page: https://sihanxu.me/nepa

### GPT解析

### 总结

本文提出了一种名为Next-Embedding Predictive Autoregression (NEPA)的自监督视觉学习方法，从学习表征转向学习模型，通过训练模型基于过去的嵌入预测未来的嵌入来实现视觉任务。

### 背景

受自然语言生成式预训练成功的启发，研究者思考同样的原则是否能产生强大的自监督视觉学习者。

### 目的

探索一种简单、可扩展且可能具有模态通用性的自监督视觉学习方法，从传统的学习表征转变为学习模型。

### 方法

提出NEPA方法，使用因果掩码和停止梯度训练模型基于过去的嵌入预测未来的嵌入，采用简单Transformer在ImageNet-1k上进行预训练，仅以嵌入预测为学习目标，无需像素重建、离散标记、对比损失或任务特定头。

### 主要发现

仅使用嵌入预测作为学习目标的简单Transformer在ImageNet-1K上表现良好，ViT-B和ViT-L骨干网络在微调后分别达到83.8%和85.3%的top-1准确率，并在ADE20K的语义分割任务中有效迁移。

### 结论

从嵌入进行生成式预训练为视觉自监督学习提供了一种简单、可扩展且可能具有模态通用性的替代方案。

### 翻译

受自然语言生成式预训练成功的启发，我们思考同样的原则是否能产生强大的自监督视觉学习者。我们不是训练模型输出特征供下游使用，而是训练它们生成嵌入来直接执行预测任务。这项工作探索了从学习表征到学习模型的这种转变。具体来说，模型学习基于过去的嵌入预测未来的嵌入，使用因果掩码和停止梯度，我们称之为Next-Embedding Predictive Autoregression (NEPA)。我们证明，一个在ImageNet-1k上仅以嵌入预测为唯一学习目标进行预训练的简单Transformer是有效的 - 不需要像素重建、离散标记、对比损失或任务特定头。这种公式保留了架构的简单性和可扩展性，不需要额外的设计复杂性。NEPA在各项任务上都取得了良好的结果，使用ViT-B和ViT-L骨干网络在微调后在ImageNet-1K上分别达到83.8%和85.3%的top-1准确率，并在ADE20K上的语义分割任务中有效迁移。我们相信，从嵌入进行生成式预训练为视觉自监督学习提供了一种简单、可扩展且可能具有模态通用性的替代方案。


### 论文摘要

Inspired by the success of generative pretraining in natural language, we ask whether the same principles can yield strong self-supervised visual learners. Instead of training models to output features for downstream use, we train them to generate embeddings to perform predictive tasks directly. This work explores such a shift from learning representations to learning models. Specifically, models learn to predict future patch embeddings conditioned on past ones, using causal masking and stop gradient, which we refer to as Next-Embedding Predictive Autoregression (NEPA). We demonstrate that a simple Transformer pretrained on ImageNet-1k with next embedding prediction as its sole learning objective is effective - no pixel reconstruction, discrete tokens, contrastive loss, or task-specific heads. This formulation retains architectural simplicity and scalability, without requiring additional design complexity. NEPA achieves strong results across tasks, attaining 83.8% and 85.3% top-1 accuracy on ImageNet-1K with ViT-B and ViT-L backbones after fine-tuning, and transferring effectively to semantic segmentation on ADE20K. We believe generative pretraining from embeddings provides a simple, scalable, and potentially modality-agnostic alternative to visual self-supervised learning.

---

## 44. Bridging Modalities and Transferring Knowledge: Enhanced Multimodal Understanding and Recognition

**论文链接:** [http://arxiv.org/abs/2512.20501v1](http://arxiv.org/abs/2512.20501v1)

**作者:** Gorjan Radevski

**发布时间:** 2025-12-23

**备注:** Ph.D. manuscript; Supervisors/Mentors: Marie-Francine Moens and Tinne Tuytelaars

### GPT解析

### 总结

该研究探索多模态对齐、翻译、融合和迁移技术，旨在提升机器对复杂输入的理解能力。论文分为五个章节，分别针对多模态机器学习中的不同挑战提出解决方案。

### 背景

多模态机器学习面临理解复杂输入的挑战，需要有效处理不同类型数据之间的关系和转换。

### 目的

增强机器对复杂多模态输入的理解能力，开发新的方法来处理空间语言、医学文本、知识图谱和动作识别等领域的问题。

### 方法

1. 第3章：开发空间推理BERT模型，将文本空间关系转换为视觉表示；2. 第4章：提出基于医学术语空间共现的损失函数，实现医学文本到解剖图谱的映射；3. 第5章：构建自然语言到实体和谓词的基准测试，解决文本提取的歧义问题；4. 第6章：提出融合视频帧和目标检测表示的方法，用于组合动作识别；5. 第7章：研究多模态知识迁移技术，使RGB模型能够模仿多模态融合能力。

### 主要发现

1. 空间推理BERT能有效解码空间语言为视觉表示；2. 基于空间共现的损失函数显著提高了医学文本的可导航性；3. 构建的知识图谱基准测试能提供更清晰、可操作的见解；4. 多模态融合方法提高了动作识别的鲁棒性和准确性；5. 多模态知识迁移能在保持性能的同时减少计算需求。

### 结论

这些研究贡献推进了空间语言理解、医学文本解释、知识图谱丰富化和动作识别的方法论，增强了计算系统处理复杂多模态输入的能力，为各种应用场景提供了新的可能性。

### 翻译

多模态对齐：建立不同模态数据之间的对应关系；多模态翻译：将一种模态的信息转换为另一种模态；多模态融合：将不同模态的信息整合处理；多模态迁移：将一种模态学到的知识应用到另一种模态；空间推理BERT：用于处理空间关系的BERT模型；解剖图谱：人体解剖结构的可视化表示；知识图谱：实体及其关系的语义网络表示；自我中心动作识别：从第一人称视角识别动作。


### 论文摘要

This manuscript explores multimodal alignment, translation, fusion, and transference to enhance machine understanding of complex inputs. We organize the work into five chapters, each addressing unique challenges in multimodal machine learning.   Chapter 3 introduces Spatial-Reasoning Bert for translating text-based spatial relations into 2D arrangements between clip-arts. This enables effective decoding of spatial language into visual representations, paving the way for automated scene generation aligned with human spatial understanding.   Chapter 4 presents a method for translating medical texts into specific 3D locations within an anatomical atlas. We introduce a loss function leveraging spatial co-occurrences of medical terms to create interpretable mappings, significantly enhancing medical text navigability.   Chapter 5 tackles translating structured text into canonical facts within knowledge graphs. We develop a benchmark for linking natural language to entities and predicates, addressing ambiguities in text extraction to provide clearer, actionable insights.   Chapter 6 explores multimodal fusion methods for compositional action recognition. We propose a method fusing video frames and object detection representations, improving recognition robustness and accuracy.   Chapter 7 investigates multimodal knowledge transference for egocentric action recognition. We demonstrate how multimodal knowledge distillation enables RGB-only models to mimic multimodal fusion-based capabilities, reducing computational requirements while maintaining performance.   These contributions advance methodologies for spatial language understanding, medical text interpretation, knowledge graph enrichment, and action recognition, enhancing computational systems' ability to process complex, multimodal inputs across diverse applications.

---

## 45. 论文ID: 2512.19871v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19871v1.json'

---

