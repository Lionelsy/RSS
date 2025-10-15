# 今日论文推荐 - 2025-10-15

共 61 篇论文

---

## 1. VideoLucy: Deep Memory Backtracking for Long Video Understanding

**论文链接:** [http://arxiv.org/abs/2510.12422v1](http://arxiv.org/abs/2510.12422v1)

**作者:** Jialong Zuo, Yongtai Deng, Lingdong Kong, Jingkang Yang, Rui Jin, Yiwei Zhang, Nong Sang, Liang Pan, Ziwei Liu, Changxin Gao

**发布时间:** 2025-10-14

**备注:** NeurIPS-2025 Accepted Paper

### GPT解析

### 总结

VideoLucy是一种深度记忆回溯框架，用于解决长视频理解中的时序上下文捕捉和关键信息保留问题，通过分层记忆结构和智能体迭代回溯机制实现优越性能。

### 背景

基于智能体的系统利用大型语言模型进行关键信息检索和整合已成为长视频理解的有前景的方法，但面临两大挑战：难以捕捉连续帧的时序上下文，以及稀疏帧采样可能导致丢弃关键信息。

### 目的

克服现有长视频理解系统的局限性，提出VideoLucy框架以有效捕捉时序上下文并保留关键信息。

### 方法

VideoLucy采用受人类从粗到细回忆过程启发的分层记忆结构，具有渐进粒度，在不同层次深度明确定义记忆的细节级别和时序范围。通过基于智能体的迭代回溯机制，系统性地挖掘视频范围内、问题相关的深度记忆，直到收集到足够信息提供自信答案。

### 主要发现

大量实验证明VideoLucy的优越性，基于开源模型构建的VideoLucy在多个长视频理解基准上显著优于最先进方法，性能甚至超过了最新的专有模型如GPT-4o。

### 结论

VideoLucy通过创新的深度记忆回溯框架有效解决了长视频理解中的关键挑战，同时引入EgoMem基准用于全面评估模型能力，代码和数据集将公开。

### 翻译

最近的研究表明，利用大型语言模型进行关键信息检索和整合的智能体系统已成为长视频理解的一种有前景的方法。然而，这些系统面临两大挑战。首先，它们通常在单帧上进行建模和推理，难以捕捉连续帧的时序上下文。其次，为降低密集帧级标注的成本，它们采用稀疏帧采样，这可能会丢弃关键信息。为克服这些局限性，我们提出了VideoLucy，一种用于长视频理解的深度记忆回溯框架。受人类从粗到细回忆过程的启发，VideoLucy采用具有渐进粒度的分层记忆结构。该结构在不同层次深度明确定义了记忆的细节级别和时序范围。通过基于智能体的迭代回溯机制，VideoLucy系统性地挖掘视频范围内、问题相关的深度记忆，直到收集到足够信息以提供自信的答案。这种设计能够有效理解连续帧的时序关系，同时保留关键细节。此外，我们引入了EgoMem，一个用于长视频理解的新基准。EgoMem旨在全面评估模型理解随时间展开的复杂事件和捕捉极长视频中细粒度细节的能力。大量实验证明了VideoLucy的优越性。基于开源模型构建的VideoLucy在多个长视频理解基准上显著优于最先进的方法，性能甚至超过了最新的专有模型如GPT-4o。我们的代码和数据集将在https://videolucy.github.io公开。


### 论文摘要

Recent studies have shown that agent-based systems leveraging large language models (LLMs) for key information retrieval and integration have emerged as a promising approach for long video understanding. However, these systems face two major challenges. First, they typically perform modeling and reasoning on individual frames, struggling to capture the temporal context of consecutive frames. Second, to reduce the cost of dense frame-level captioning, they adopt sparse frame sampling, which risks discarding crucial information. To overcome these limitations, we propose VideoLucy, a deep memory backtracking framework for long video understanding. Inspired by the human recollection process from coarse to fine, VideoLucy employs a hierarchical memory structure with progressive granularity. This structure explicitly defines the detail level and temporal scope of memory at different hierarchical depths. Through an agent-based iterative backtracking mechanism, VideoLucy systematically mines video-wide, question-relevant deep memories until sufficient information is gathered to provide a confident answer. This design enables effective temporal understanding of consecutive frames while preserving critical details. In addition, we introduce EgoMem, a new benchmark for long video understanding. EgoMem is designed to comprehensively evaluate a model's ability to understand complex events that unfold over time and capture fine-grained details in extremely long videos. Extensive experiments demonstrate the superiority of VideoLucy. Built on open-source models, VideoLucy significantly outperforms state-of-the-art methods on multiple long video understanding benchmarks, achieving performance even surpassing the latest proprietary models such as GPT-4o. Our code and dataset will be made publicly at https://videolucy.github.io

---

## 2. Learning to Recognize Correctly Completed Procedure Steps in Egocentric Assembly Videos through Spatio-Temporal Modeling

**论文链接:** [http://arxiv.org/abs/2510.12385v1](http://arxiv.org/abs/2510.12385v1)

**作者:** Tim J. Schoonbeek, Shao-Hsuan Hung, Dan Lehman, Hans Onvlee, Jacek Kustra, Peter H. N. de With, Fons van der Sommen

**发布时间:** 2025-10-14

**备注:** 26 pages, 7 figures and 5 tables in the main paper and one figure and  table in the appendix. To be published in Computer Vision and Image  Understanding

### GPT解析

### 总结

本文提出了STORM-PSR模型，一种双流框架的程序步骤识别方法，通过结合空间和时间特征，有效解决了物体部分遮挡情况下的步骤识别问题。

### 背景

现有的程序步骤识别模型仅依靠检测单个视频帧中的装配对象状态，忽略了时间特征，导致模型鲁棒性和准确性有限，尤其在物体部分遮挡时表现不佳。

### 目的

克服现有方法的局限性，提出一种能够处理部分遮挡情况下的程序步骤识别方法，提高识别准确性和鲁棒性。

### 方法

提出STORM-PSR（时空遮挡弹性建模程序步骤识别）双流框架：装配状态检测流在物体无遮挡时有效工作；时空流捕捉空间和时间特征，即使在部分遮挡下也能识别步骤完成情况。时空流包含使用弱监督方法预训练的空间编码器和基于transformer的时间编码器。

### 主要发现

在MECCANO和IndustReal数据集上评估，与之前方法相比，分别减少了11.2%和26.1%的实际和预测装配步骤完成之间的平均延迟。这种改进主要由时空流实现，它不依赖物体的无遮挡视图。

### 结论

STORM-PSR能有效处理部分遮挡情况下的程序步骤识别问题，显著提高了识别准确性和及时性。相关代码和数据集已公开可用。

### 翻译

程序步骤识别旨在识别视频中程序任务中所有正确完成的步骤及其顺序。现有的最先进模型仅依靠检测单个视频帧中的装配对象状态。通过忽略时间特征，模型的鲁棒性和准确性受到限制，特别是当物体部分遮挡时。为克服这些限制，我们提出了用于程序步骤识别的时空遮挡弹性建模（STORM-PSR），这是一个用于程序步骤识别的双流框架，同时利用空间和时间特征。装配状态检测流在物体无遮挡视图下有效工作，而时空流捕捉空间和时间特征，即使在部分遮挡下也能识别步骤完成情况。该流包括一个空间编码器，使用新颖的弱监督方法预训练以捕获有意义的空间表示，以及一个基于transformer的时间编码器，学习这些空间特征随时间的关系。STORM-PSR在MECCANO和IndustReal数据集上进行了评估，与之前的方法相比，分别减少了11.2%和26.1%的实际和预测装配步骤完成之间的平均延迟。我们证明这种延迟减少是由时空流驱动的，它不依赖物体的无遮挡视图来推断完成的步骤。STORM-PSR的代码以及新标注的MECCANO标签已在https://timschoonbeek.github.io/stormpsr公开提供。


### 论文摘要

Procedure step recognition (PSR) aims to identify all correctly completed steps and their sequential order in videos of procedural tasks. The existing state-of-the-art models rely solely on detecting assembly object states in individual video frames. By neglecting temporal features, model robustness and accuracy are limited, especially when objects are partially occluded. To overcome these limitations, we propose Spatio-Temporal Occlusion-Resilient Modeling for Procedure Step Recognition (STORM-PSR), a dual-stream framework for PSR that leverages both spatial and temporal features. The assembly state detection stream operates effectively with unobstructed views of the object, while the spatio-temporal stream captures both spatial and temporal features to recognize step completions even under partial occlusion. This stream includes a spatial encoder, pre-trained using a novel weakly supervised approach to capture meaningful spatial representations, and a transformer-based temporal encoder that learns how these spatial features relate over time. STORM-PSR is evaluated on the MECCANO and IndustReal datasets, reducing the average delay between actual and predicted assembly step completions by 11.2% and 26.1%, respectively, compared to prior methods. We demonstrate that this reduction in delay is driven by the spatio-temporal stream, which does not rely on unobstructed views of the object to infer completed steps. The code for STORM-PSR, along with the newly annotated MECCANO labels, is made publicly available at https://timschoonbeek.github.io/stormpsr .

---

## 3. State Space Prompting via Gathering and Spreading Spatio-Temporal Information for Video Understanding

**论文链接:** [http://arxiv.org/abs/2510.12160v1](http://arxiv.org/abs/2510.12160v1)

**作者:** Jiahuan Zhou, Kai Zhu, Zhenyu Cui, Zichen Liu, Xu Zou, Gang Hua

**发布时间:** 2025-10-14

### GPT解析

### 总结

该研究提出了一种状态空间提示方法，通过结合帧内和帧间提示来有效捕捉和传播视频中的时空信息，显著提升了视频分类性能。

### 背景

预训练的状态空间模型在视频分类方面展现出巨大潜力，它们以线性复杂度顺序压缩视频中的视觉标记，提高处理效率的同时保持高性能。提示学习被用于将这些强大模型高效适配到下游任务。

### 目的

解决顺序压缩的视觉提示标记无法充分捕捉视频时空上下文信息的问题，以增强状态压缩模型中空间信息在帧内的传播以及时间信息在帧间的提取。

### 方法

提出状态空间提示方法，结合帧内和帧间提示聚合和传播视频中的关键时空信息。具体包括帧内聚合模块和帧间扩散模块，通过自适应平衡和压缩帧内及帧间的关键时空信息，以互补方式有效传播判别性信息。

### 主要发现

在四个视频基准数据集上的实验表明，该方法平均比现有最先进方法高出2.76%，同时减少了微调参数的开销。

### 结论

该方法通过结合帧内和帧间提示，有效解决了状态空间模型在视频理解中时空信息捕捉不足的问题，在保持高性能的同时提高了处理效率并减少了微调参数。

### 翻译

最近，预训练的状态空间模型在视频分类方面显示出巨大潜力，它们以线性复杂度顺序压缩视频中的视觉标记，从而提高视频数据的处理效率同时保持高性能。为了将强大的预训练模型应用于下游任务，提示学习被提出，只需少量微调参数即可实现高效的下游任务适应。然而，顺序压缩的视觉提示标记无法捕捉视频中的空间和时间上下文信息，从而限制了状态压缩模型内空间信息在视频帧内的有效传播以及帧间时间信息的提取和判别性信息的提取。为解决上述问题，我们提出了一种用于视频理解的状态空间提示方法，该方法结合帧内和帧间提示来聚合和传播视频中的关键时空信息。具体来说，设计了帧内聚合模块来聚合每个帧内的空间关键信息。此外，设计了帧间扩散模块来传播不同帧间的判别性时空信息。通过自适应平衡和压缩帧内及帧间的关键时空信息，我们的方法以互补方式有效传播视频中的判别性信息。在四个视频基准数据集上的大量实验验证了我们的方法平均比现有最先进方法高出2.76%，同时减少了微调参数的开销。


### 论文摘要

Recently, pre-trained state space models have shown great potential for video classification, which sequentially compresses visual tokens in videos with linear complexity, thereby improving the processing efficiency of video data while maintaining high performance. To apply powerful pre-trained models to downstream tasks, prompt learning is proposed to achieve efficient downstream task adaptation with only a small number of fine-tuned parameters. However, the sequentially compressed visual prompt tokens fail to capture the spatial and temporal contextual information in the video, thus limiting the effective propagation of spatial information within a video frame and temporal information between frames in the state compression model and the extraction of discriminative information. To tackle the above issue, we proposed a State Space Prompting (SSP) method for video understanding, which combines intra-frame and inter-frame prompts to aggregate and propagate key spatiotemporal information in the video. Specifically, an Intra-Frame Gathering (IFG) module is designed to aggregate spatial key information within each frame. Besides, an Inter-Frame Spreading (IFS) module is designed to spread discriminative spatio-temporal information across different frames. By adaptively balancing and compressing key spatio-temporal information within and between frames, our SSP effectively propagates discriminative information in videos in a complementary manner. Extensive experiments on four video benchmark datasets verify that our SSP significantly outperforms existing SOTA methods by 2.76% on average while reducing the overhead of fine-tuning parameters.

---

## 4. Task-Specific Dual-Model Framework for Comprehensive Traffic Safety Video Description and Analysis

**论文链接:** [http://arxiv.org/abs/2510.11907v1](http://arxiv.org/abs/2510.11907v1)

**作者:** Blessing Agyei Kyem, Neema Jakisa Owor, Andrews Danyo, Joshua Kofi Asamoah, Eugene Denteh, Tanner Muturi, Anthony Dontoh, Yaw Adu-Gyamfi, Armstrong Aboah

**发布时间:** 2025-10-13

**备注:** This paper was accepted at ICCV 2025

### GPT解析

### 总结

本研究提出了一种双模型框架，结合VideoLLaMA和Qwen2.5-VL的优势，通过分离训练字幕生成和视觉问答任务来最小化任务干扰，提高交通安全分析能力。实验证明该方法在AI城市挑战赛中表现优异，分离训练策略优于联合训练。

### 背景

交通安全分析需要复杂的视频理解技术，以捕获细粒度的行为模式并生成全面的描述用于事故预防。

### 目的

开发一种能够有效进行交通安全分析的框架，通过结合不同模型的优势提高视频理解和分析能力，从而更好地捕捉交通行为模式和预防事故。

### 方法

提出独特的双模型框架，战略性地利用VideoLLaMA和Qwen2.5-VL的互补优势。核心思路是分离字幕生成和视觉问答任务的训练，以最小化任务干扰并使每个模型能够更有效地专业化。通过在WTS数据集上进行大量实验评估该方法。

### 主要发现

VideoLLaMA在时间推理方面特别有效，达到1.1001的CIDEr分数；Qwen2.5-VL在视觉理解方面表现出色，VQA准确率达到60.80%；该方法在2025年AI城市挑战赛第二赛道中达到45.7572的S2分数，排名第10位；分离训练策略比联合训练在VQA准确率上提高8.6%，同时保持字幕生成质量。

### 结论

通过分离训练策略，双模型框架能够有效地结合不同模型的专长，提高交通安全分析的性能。VideoLLaMA擅长时间推理，而Qwen2.5-VL在视觉理解方面表现优异，为交通安全视频分析提供了有效的解决方案。

### 翻译

交通安全分析需要复杂的视频理解来捕获细粒度的行为模式并生成全面的描述以预防事故。在这项工作中，我们提出了一种独特的双模型框架，通过针对特定任务的优化，战略性地利用VideoLLaMA和Qwen2.5-VL的互补优势来解决这一问题。我们方法的核心见解是，分离字幕生成和视觉问答任务的训练可以最小化任务干扰，并使每个模型能够更有效地专业化。实验结果表明，VideoLLaMA在时间推理方面特别有效，达到1.1001的CIDEr分数，而Qwen2.5-VL在视觉理解方面表现出色，VQA准确率为60.80%。通过在WTS数据集上的大量实验，我们的方法在2025年AI城市挑战赛第二赛道中实现了45.7572的S2分数，在挑战排行榜上排名第10位。消融研究验证了我们的分离训练策略在VQA准确率上比联合训练提高了8.6%，同时保持了字幕生成质量。


### 论文摘要

Traffic safety analysis requires complex video understanding to capture fine-grained behavioral patterns and generate comprehensive descriptions for accident prevention. In this work, we present a unique dual-model framework that strategically utilizes the complementary strengths of VideoLLaMA and Qwen2.5-VL through task-specific optimization to address this issue. The core insight behind our approach is that separating training for captioning and visual question answering (VQA) tasks minimizes task interference and allows each model to specialize more effectively. Experimental results demonstrate that VideoLLaMA is particularly effective in temporal reasoning, achieving a CIDEr score of 1.1001, while Qwen2.5-VL excels in visual understanding with a VQA accuracy of 60.80\%. Through extensive experiments on the WTS dataset, our method achieves an S2 score of 45.7572 in the 2025 AI City Challenge Track 2, placing 10th on the challenge leaderboard. Ablation studies validate that our separate training strategy outperforms joint training by 8.6\% in VQA accuracy while maintaining captioning quality.

---

## 5. Audio-Guided Visual Perception for Audio-Visual Navigation

**论文链接:** [http://arxiv.org/abs/2510.11760v1](http://arxiv.org/abs/2510.11760v1)

**作者:** Yi Wang, Yinfeng Yu, Fuchun Sun, Liejun Wang, Wendong Zheng

**发布时间:** 2025-10-13

**备注:** Main paper (6 pages). Accepted for publication by International  Conference on Virtual Reality and Visualization 2025 (ICVRV 2025)

### GPT解析

### 总结

视听具身导航(AVN)旨在使智能体能够使用听觉线索在未知3D环境中自主导航到声源。当前方法在已知声源上表现良好，但在面对新声源时泛化能力差。AGVP框架通过跨模态对齐和区域重加权解决了这一问题，提高了导航效率和鲁棒性。

### 背景

当前的视听具身导航方法在分布内的声源上表现良好，但在跨声源泛化方面表现较差，当遇到未听过的声音或未见过的环境时，导航成功率大幅下降，搜索路径变得过长。

### 目的

解决当前AVN方法在跨声源泛化方面的局限性，提高智能体在面对新声源时的导航效率和鲁棒性。

### 方法

提出AGVP框架，该框架首先通过音频自注意力提取全局听觉上下文，然后使用此上下文作为查询来引导视觉特征注意力，在特征级别突出显示与声源相关的区域，随后进行时间建模和策略优化。设计以可解释的跨模态对齐和区域重加权为中心，减少对特定声学指纹的依赖。

### 主要发现

AGVP框架通过将声音从策略可记忆的声学指纹线索转换为空间引导，解决了当前方法中缺乏听觉信号与相应视觉区域之间明确对齐机制的问题，避免了策略在训练期间记忆虚假的'声学指纹-场景'相关性。

### 结论

AGVP框架提高了导航效率和鲁棒性，同时对先前未听过的声音实现了跨场景的优越泛化，减少了方法对特定声学指纹的依赖。

### 翻译

视听具身导航旨在使智能体能够使用听觉线索在未知3D环境中自主导航到声源。虽然当前的AVN方法在分布内的声源上表现出色，但在跨声源泛化方面表现不佳：当智能体遇到未听过的声音或未见过的环境时，导航成功率大幅下降，搜索路径变得过长。这种限制源于缺乏听觉信号与相应视觉区域之间的明确对齐机制。策略倾向于在训练期间记忆虚假的'声学指纹-场景'相关性，当遇到新的声源时导致盲目探索。为解决此问题，我们提出了AGVP框架，将声音从策略可记忆的声学指纹线索转换为空间引导。该框架首先通过音频自注意力提取全局听觉上下文，然后使用此上下文作为查询来引导视觉特征注意力，在特征级别突出显示与声源相关的区域。随后进行时间建模和策略优化。这种以可解释的跨模态对齐和区域重加权为中心的设计，减少了对特定声学指纹的依赖。实验结果表明，AGVP提高了导航效率和鲁棒性，同时对先前未听过的声音实现了跨场景的优越泛化。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决音频视觉导航(AVN)中的跨源泛化能力不足问题。当前方法在训练时听过的声音源上表现很好(成功率超过95%)，但在遇到未听过的声音源或未见过的环境时，导航成功率急剧下降，搜索路径变得过长。这个问题在现实中非常重要，因为现实世界中的声音源是多样化的，我们无法预训练智能体处理所有可能的声音。在紧急情况下(如火灾中求救声)，智能体需要能够有效定位未知声音源，提高跨场景泛化能力对于构建真正实用的自主导航系统至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了当前AVN方法的局限性：它们在策略级别连接视觉和听觉特征，缺乏声音源相关视觉区域与听觉信号之间的明确对齐，导致策略网络倾向于记忆'声纹-场景'的虚假相关性。作者从人类导航行为中获得灵感：在视觉受阻时，人类会先转向声音方向，锁定大致方向，然后关注声音可能出现的区域。基于此，作者设计了'声音优先，视觉跟随'的多模态融合机制。该方法借鉴了SoundSpaces平台、Transformer架构中的自注意力机制和引导注意力机制，以及GRU进行时间建模，同时参考了现有音频视觉导航方法如AV-WaN、SAVi等的优缺点。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "AGVP框架的核心思想是将声音从策略可记忆的声纹线索转变为空间引导。在复杂视觉推理之前，使用音频上下文重新校准视觉特征图，突出显示与声音源最相关的区域，使音频决定'看哪里'，而视觉决定'如何看'。整体流程分为三阶段：1)观察阶段：智能体通过传感器获取视觉(深度图或RGB图像)和听觉(双通道频谱图)输入；2)观察编码阶段：音频特征通过自注意力构建全局上下文，然后作为查询引导视觉特征注意力，突出声音源相关区域，随后通过GRU进行时间建模；3)策略更新阶段：基于PPO的Actor-Critic头根据GRU隐藏状态生成动作分布和状态值，完成从感知到决策的闭环。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)'声音优先，视觉跟随'的多模态融合机制，将融合从后端策略级别提升到感知特征级别；2)音频引导的视觉特征重新加权，使用音频上下文引导视觉注意力；3)通过可解释的跨模态对齐和区域重新加权，减少对特定声纹的依赖。相比之前工作，AGVP在特征级别实现明确的音频视觉对齐，而非传统方法在策略级别的简单连接；不再依赖'声纹-场景'的记忆映射，而是将声音转变为空间引导；在未听过的声音源任务上表现显著优于现有方法，如Replica数据集上实现66.5%的成功率，比之前最佳方法提高约15个百分点。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'AGVP框架通过在特征级别实现音频到视觉的明确对齐，将声音从可记忆的声纹线索转变为空间引导，显著提升了音频视觉导航系统在未知声音源和未见环境中的泛化能力和导航效率。'}


### 论文摘要

Audio-Visual Embodied Navigation aims to enable agents to autonomously navigate to sound sources in unknown 3D environments using auditory cues. While current AVN methods excel on in-distribution sound sources, they exhibit poor cross-source generalization: navigation success rates plummet and search paths become excessively long when agents encounter unheard sounds or unseen environments. This limitation stems from the lack of explicit alignment mechanisms between auditory signals and corresponding visual regions. Policies tend to memorize spurious \enquote{acoustic fingerprint-scenario} correlations during training, leading to blind exploration when exposed to novel sound sources. To address this, we propose the AGVP framework, which transforms sound from policy-memorable acoustic fingerprint cues into spatial guidance. The framework first extracts global auditory context via audio self-attention, then uses this context as queries to guide visual feature attention, highlighting sound-source-related regions at the feature level. Subsequent temporal modeling and policy optimization are then performed. This design, centered on interpretable cross-modal alignment and region reweighting, reduces dependency on specific acoustic fingerprints. Experimental results demonstrate that AGVP improves both navigation efficiency and robustness while achieving superior cross-scenario generalization on previously unheard sounds.

---

## 6. SPORTS: Simultaneous Panoptic Odometry, Rendering, Tracking and Segmentation for Urban Scenes Understanding

**论文链接:** [http://arxiv.org/abs/2510.12749v1](http://arxiv.org/abs/2510.12749v1)

**作者:** Zhiliu Yang, Jinyu Dai, Jianyuan Zhang, Zhu Yang

**发布时间:** 2025-10-14

**备注:** Accepted by IEEE Transactions on Multimedia

### GPT解析

### 总结

本文提出了一个名为SPORTS的新型框架，通过紧密集成视频全景分割、视觉里程计和场景渲染任务，实现整体场景理解，解决了现有方法中的分割不足、动态物体干扰、传感器数据稀疏和视角限制等问题。

### 背景

场景感知、理解和模拟是具身AI代理的基础技术，但现有解决方案仍存在分割不足、动态物体干扰、传感器数据稀疏和视角限制等挑战。

### 目的

设计一个统一的框架，通过整合视频全景分割、视觉里程计和场景渲染任务，实现更准确的整体场景理解。

### 方法

1) VPS部分：设计基于自适应注意力的几何融合机制，通过引入姿态、深度和光流模态对齐跨帧特征，并集成后匹配策略改进身份跟踪；2) VO部分：结合VPS的全景分割结果和光流图，提高动态物体置信度估计，增强相机姿态估计精度和深度图生成完整性；3) SR部分：将稀疏点云转换为神经场，合成高保真RGB视图和双重视图。

### 主要发现

在三个公共数据集上的实验表明，基于注意力的特征融合在里程计、跟踪、分割和新视角合成任务上优于大多数现有最先进方法。

### 结论

SPORTS框架通过迭代和统一的视角整合多种任务，有效解决了场景理解中的多个挑战性问题，提升了整体性能。

### 翻译

场景感知、理解和模拟是具身AI代理的基础技术，而现有解决方案仍然容易受到分割不足、动态物体干扰、传感器数据稀疏和视角限制等问题的影响。本文提出了一种名为SPORTS的新型框架，通过将视频全景分割、视觉里程计和场景渲染任务紧密集成到一个迭代和统一的视角中，实现整体场景理解。首先，VPS设计了一种基于自适应注意力的几何融合机制，通过引入姿态、深度和光流模态来对齐跨帧特征，自动调整不同解码阶段的特征图，并集成了后匹配策略以改进身份跟踪。在VO中，VPS的全景分割结果与光流图相结合，提高了动态物体的置信度估计，通过基于学习的方法增强了相机姿态估计的精度和深度图生成的完整性。此外，SR的点渲染受益于VO，将稀疏点云转换为神经场，以合成高保真的RGB视图和双重视图。在三个公共数据集上的大量实验证明，我们的基于注意力的特征融合在里程计、跟踪、分割和新视角合成任务上优于大多数现有的最先进方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有场景感知、理解和模拟技术中的四个关键问题：分割缺陷、动态物体干扰、传感器数据稀疏和视角限制。这些问题在现实中非常重要，因为随着自动驾驶车辆、四足机器人和人形机器人的普及，对城市场景的整体理解能力对这些智能体执行感知、定位和碰撞避免等任务至关重要。此外，整体场景理解可用于创建城市环境的数字孪生，作为智能体学习和验证意外情况的模拟平台，从而以较低成本提高自动驾驶安全性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有方法已开始解决整体场景理解中的信息孤岛问题，但VO和VPS性能仍需改进。他们注意到动态物体处理和稀疏点云重建是关键挑战。设计上，作者借鉴了Video K-Net的核学习机制、PVO的集成方法、DROID-SLAM的优化策略以及READ的点云渲染方法，但通过引入基于注意力的自适应几何融合机制和后匹配策略进行创新，解决了现有方法的局限性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是提出名为SPORTS的框架，通过紧密集成视频全景分割(VPS)、视觉里程计(VO)和场景渲染(SR)任务，实现迭代统一的城市场景理解。整体流程为：1)输入单目视频序列；2)VPS模块利用基于注意力的几何融合机制对齐跨帧特征，并加入后匹配策略提高跟踪质量；3)VO模块利用VPS结果增强动态物体置信度估计，提高姿态估计精度；4)SR模块将高精度姿态保证的稀疏点云转换为神经场，合成高保真场景；5)输出稀疏点云地图、相机姿态和合成的新场景。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次统一视觉里程计、渲染、物体跟踪和全景分割任务的框架；2)提出基于注意力的几何融合机制和后匹配策略，提高分割和跟踪质量3.07%；3)提出两阶段全景感知流感知深度传播模块，增强基于学习的视觉里程计；4)利用基础模型创建更多评估数据集，验证泛化能力。相比之前工作，SPORTS更充分地考虑了相邻帧特征，采用更先进的解码网络，解决了长序列中的误差传播问题，并更好地区分了静止可移动物体和真正移动的物体。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SPORTS通过统一视觉里程计、渲染、物体跟踪和全景分割任务，并提出基于注意力的自适应几何融合机制和后匹配策略，实现了对城市场景的高效理解和精确重建，显著提升了动态环境下的感知、定位和场景渲染性能。'}


### 论文摘要

The scene perception, understanding, and simulation are fundamental techniques for embodied-AI agents, while existing solutions are still prone to segmentation deficiency, dynamic objects' interference, sensor data sparsity, and view-limitation problems. This paper proposes a novel framework, named SPORTS, for holistic scene understanding via tightly integrating Video Panoptic Segmentation (VPS), Visual Odometry (VO), and Scene Rendering (SR) tasks into an iterative and unified perspective. Firstly, VPS designs an adaptive attention-based geometric fusion mechanism to align cross-frame features via enrolling the pose, depth, and optical flow modality, which automatically adjust feature maps for different decoding stages. And a post-matching strategy is integrated to improve identities tracking. In VO, panoptic segmentation results from VPS are combined with the optical flow map to improve the confidence estimation of dynamic objects, which enhances the accuracy of the camera pose estimation and completeness of the depth map generation via the learning-based paradigm. Furthermore, the point-based rendering of SR is beneficial from VO, transforming sparse point clouds into neural fields to synthesize high-fidelity RGB views and twin panoptic views. Extensive experiments on three public datasets demonstrate that our attention-based feature fusion outperforms most existing state-of-the-art methods on the odometry, tracking, segmentation, and novel view synthesis tasks.

---

## 7. Gaussian Semantic Field for One-shot LiDAR Global Localization

**论文链接:** [http://arxiv.org/abs/2510.12101v1](http://arxiv.org/abs/2510.12101v1)

**作者:** Pengyu Yin, Shenghai Yuan, Haozhi Cao, Xingyu Ji, Ruofei Bai, Siyu Chen, Lihua Xie

**发布时间:** 2025-10-14

### GPT解析

### 总结

提出了一种基于轻量级三层场景图的一次性激光雷达全局定位算法，具有语义消歧能力。

### 背景

基于地标语义注册的全局定位方法相比纯几何方法已经显示出有前景的性能提升，但地标可能是重复的且可能误导对应关系的建立。

### 目的

通过使用高斯过程群体学习到的连续函数来建模语义分布，缓解地标重复和误导性的问题。

### 方法

将连续函数作为中间层插入到物体层和度量-语义层之间，形成三层3D场景图，作为一次性定位的轻量级高性能后端。

### 主要发现

与离散语义标签相比，连续函数能够捕获更细粒度的地理语义信息，并为对应关系建立提供更详细的度量信息。

### 结论

将全局定位管道命名为Outram-GSF（高斯语义场），并在公开可用数据集上进行了广泛实验，验证了其与当前最先进方法相比的优越性能。

### 翻译

我们提出了一种基于轻量级三层场景图的一次性激光雷达全局定位算法，具有语义消歧能力。虽然基于地标语义注册的方法与纯几何方法相比已经在全局定位中显示出有前景的性能提升，但地标可能是重复的且可能误导对应关系的建立。我们提出通过使用从高斯过程群体学习到的连续函数来建模语义分布来缓解这一问题。与离散语义标签相比，连续函数能够捕获更细粒度的地理语义信息，并为对应关系建立提供更详细的度量信息。我们将这个连续函数作为中间层插入到物体层和度量-语义层之间，形成三层3D场景图，作为一次性定位的轻量级高性能后端。我们将我们的全局定位管道命名为Outram-GSF（高斯语义场），并在公开可用数据集上进行了广泛实验，验证了其与当前最先进方法相比的优越性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决激光雷达全局定位中的语义歧义问题，即在具有重复结构的环境（如城市街道、停车场）中，传统方法难以区分相似但位置不同的地标。这个问题在现实中非常重要，因为精确的全局定位是自动驾驶和机器人导航的基础能力，而一次性定位方法对于机器人快速重新定位至关重要，特别是在GPS信号弱或不可用的环境中。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有基于场景图的全局定位方法主要依赖语义簇的质心和拓扑连接，忽略了簇内丰富的空间语义分布。他们借鉴了3D场景图概念、Outram的分层搜索思想以及高斯过程建模方法，在此基础上创新性地引入高斯语义场作为中间层，形成三层结构。作者通过连续建模空间语义分布来区分几何相似但语义分布不同的区域，解决语义歧义问题，特别是在重复语义结构环境中提高性能。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用高斯语义场(GSF)建模连续的语义分布，而不是使用离散的语义标签，在传统3D场景图的对象层和语义点云层之间插入GSF中间层。整体流程包括：1)生成三层3D场景图，通过稀疏高斯过程创建高斯语义场层；2)使用网格探测方法生成度量语义特征；3)基于图的子结构匹配，利用Wasserstein距离进行相似性测量；4)通过最大团过程选择内点对应关系，生成最终姿态估计。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)引入高斯语义场作为3D场景图的中间层，实现连续语义分布建模；2)基于高斯过程的概率框架学习语义分布并提供相似度度量；3)三层场景图结构作为轻量级高性能定位后端；4)语义稳定性掩码处理场景变化。相比之前工作，Outram-GSF不再仅依赖实例级质心和拓扑关系，而是捕捉簇内丰富的空间语义分布，提供更详细的度量信息，在语义歧义环境中表现更好，特别是在重复结构场景中显著提高了定位性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于高斯语义场的一次性激光雷达全局定位方法，通过连续建模空间语义分布解决了传统方法在语义歧义环境中的局限性，显著提高了在重复结构场景中的定位性能。'}


### 论文摘要

We present a one-shot LiDAR global localization algorithm featuring semantic disambiguation ability based on a lightweight tri-layered scene graph. While landmark semantic registration-based methods have shown promising performance improvements in global localization compared with geometric-only methods, landmarks can be repetitive and misleading for correspondence establishment. We propose to mitigate this problem by modeling semantic distributions with continuous functions learned from a population of Gaussian processes. Compared with discrete semantic labels, the continuous functions capture finer-grained geo-semantic information and also provide more detailed metric information for correspondence establishment. We insert this continuous function as the middle layer between the object layer and the metric-semantic layer, forming a tri-layered 3D scene graph, serving as a light-weight yet performant backend for one-shot localization. We term our global localization pipeline Outram-GSF (Gaussian semantic field) and conduct a wide range of experiments on publicly available data sets, validating the superior performance against the current state-of-the-art.

---

## 8. Prompt-Guided Spatial Understanding with RGB-D Transformers for Fine-Grained Object Relation Reasoning

**论文链接:** [http://arxiv.org/abs/2510.11996v1](http://arxiv.org/abs/2510.11996v1)

**作者:** Tanner Muturi, Blessing Agyei Kyem, Joshua Kofi Asamoah, Neema Jakisa Owor, Richard Dyzinela, Andrews Danyo, Yaw Adu-Gyamfi, Armstrong Aboah

**发布时间:** 2025-10-13

**备注:** The paper was accepted at ICCV Conference 2025

### GPT解析

### 总结

本文介绍了一个专门的空间推理框架，通过将掩模尺寸嵌入输入提示中，增强模型对物体几何和布局的理解能力，并在四个特定问题类别上进行微调，最终在AI City Challenge的Track 3中排名第四，得分为73.0606，证明了该方法在现实工业环境中空间推理的有效性。

### 背景

大规模3D环境（如仓库）中的空间推理对视觉语言系统仍然是一个重大挑战，主要困难包括场景杂乱、遮挡以及需要精确的空间理解。现有模型在这样的环境中往往难以泛化，因为它们严重依赖局部外观，缺乏明确的空间基础。

### 目的

为2025年AI City Challenge的Track 3中介绍的Physical AI Spatial Intelligence Warehouse数据集引入一个专门的空间推理框架，以改善视觉语言系统在复杂3D环境中的空间推理能力。

### 方法

通过将掩模尺寸以边界框坐标的形式直接嵌入输入提示中，增强空间理解能力，使模型能够对物体几何和布局进行推理。在四个问题类别上进行微调框架：距离估计、物体计数、多选基础和空间关系推理，使用任务特定的监督。为了进一步提高与评估系统的一致性，将标准化答案附加到训练集中的GPT响应中。

### 主要发现

综合管道最终得分为73.0606，在公共排行榜上总体排名第四。这些结果表明，结构化提示增强和有针对性的优化在推进现实工业环境中的空间推理方面是有效的。

### 结论

结构化提示增强和有针对性的优化在推进现实工业环境中的空间推理方面是有效的，通过嵌入掩模尺寸和任务特定微调，可以显著提升视觉语言系统在复杂3D环境中的空间推理能力。

### 翻译

在大规模3D环境（如仓库）中进行空间推理对于视觉语言系统来说仍然是一个重大挑战，因为场景杂乱、遮挡以及需要精确的空间理解。现有模型在这样的环境中往往难以泛化，因为它们严重依赖局部外观，缺乏明确的空间基础。在这项工作中，我们为2025年AI City Challenge的Track 3中介绍的Physical AI Spatial Intelligence Warehouse数据集引入了一个专门的空间推理框架。我们的方法通过将掩模尺寸以边界框坐标的形式直接嵌入输入提示中，增强空间理解能力，使模型能够对物体几何和布局进行推理。我们在四个问题类别上微调框架：距离估计、物体计数、多选基础和空间关系推理，使用任务特定的监督。为了进一步提高与评估系统的一致性，将标准化答案附加到训练集中的GPT响应中。我们的综合管道最终得分为73.0606，在公共排行榜上总体排名第四。这些结果表明，结构化提示增强和有针对性的优化在推进现实工业环境中的空间推理方面是有效的。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决大型3D环境（如仓库）中的空间推理挑战。视觉-语言系统在这些场景中面临杂乱环境、遮挡和需要精确空间理解的问题，现有模型往往难以泛化，因为它们过度依赖局部外观而缺乏明确的空间基础。这个问题在工业环境中至关重要，因为空间推理对仓库导航、库存管理和安全监控等任务必不可少，而这些环境具有不规则结构、多样物体类型和频繁遮挡等特点，需要系统能够同时捕捉细粒度视觉细节和场景的广泛空间组织。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有视觉-语言模型在空间推理方面的局限性，特别是在处理2D图像和缺乏几何基础方面。他们分析了仓库环境的复杂性，认为需要结合对象检测和空间理解的方法。作者借鉴了SpatialBot框架，因为它已证明在空间智能方面的优越性，能够将单目深度估计生成的深度信息整合到RGB输入中。在此基础上，作者设计了提示级别的增强，将区域掩码编码为边界框坐标，提供结构化空间线索，并添加标准化答案格式确保与评估系统一致。他们还采用了LoRA微调技术来减少训练时间和内存需求。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过在提示中嵌入边界框坐标和掩码尺寸，将几何信息直接注入到视觉-语言模型的输入中，结合深度信息增强模型对空间关系的理解。整体实现流程包括：1) 基于SpatialBot的模型架构，处理RGB和深度输入；2) 提示增强，在输入中注入边界框坐标和区域ID；3) 答案标准化，添加模板后缀确保格式一致；4) 在仓库数据集上进行任务特定微调，使用LoRA技术优化训练过程。这种方法使模型能够推理物体几何和布局，提高在复杂仓库环境中的空间推理能力。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 专门针对3D工业环境的空间问答框架；2) 提示增强方法，将物体级别的几何特征嵌入提示；3) 在仓库数据集上扩展SpatialBot架构；4) 输出标准化模块确保预测一致性；5) 在AI City Challenge中取得第四名的优异表现。相比之前工作，该方法明确注入几何信息而非依赖纯2D图像，专门针对仓库环境复杂性优化，通过提示工程提供更精确的空间定位，并通过答案标准化确保输出一致性，超越了仅使用语言输入引导空间预测的传统方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过在提示中嵌入边界框坐标和深度信息，并应用答案标准化技术，显著提升了视觉-语言模型在复杂仓库环境中的细粒度空间推理能力。'}


### 论文摘要

Spatial reasoning in large-scale 3D environments such as warehouses remains a significant challenge for vision-language systems due to scene clutter, occlusions, and the need for precise spatial understanding. Existing models often struggle with generalization in such settings, as they rely heavily on local appearance and lack explicit spatial grounding. In this work, we introduce a dedicated spatial reasoning framework for the Physical AI Spatial Intelligence Warehouse dataset introduced in the Track 3 2025 AI City Challenge. Our approach enhances spatial comprehension by embedding mask dimensions in the form of bounding box coordinates directly into the input prompts, enabling the model to reason over object geometry and layout. We fine-tune the framework across four question categories namely: Distance Estimation, Object Counting, Multi-choice Grounding, and Spatial Relation Inference using task-specific supervision. To further improve consistency with the evaluation system, normalized answers are appended to the GPT response within the training set. Our comprehensive pipeline achieves a final score of 73.0606, placing 4th overall on the public leaderboard. These results demonstrate the effectiveness of structured prompt enrichment and targeted optimization in advancing spatial reasoning for real-world industrial environments.

---

## 9. REACT3D: Recovering Articulations for Interactive Physical 3D Scenes

**论文链接:** [http://arxiv.org/abs/2510.11340v2](http://arxiv.org/abs/2510.11340v2)

**作者:** Zhao Huang, Boyang Sun, Alexandros Delitzas, Jiaqi Chen, Marc Pollefeys

**发布时间:** 2025-10-13

**备注:** 8 pages

### GPT解析

### 总结

REACT3D是一个可扩展的零样本框架，将静态3D场景转换为具有一致几何形状的模拟就绪交互式副本，可直接用于各种下游任务，通过四个主要贡献实现高效处理，为具身智能研究提供了实用工具。

### 背景

交互式3D场景对具身智能日益重要，但由于注释部分分割、运动类型和运动轨迹的劳动密集型过程，现有数据集仍然有限。

### 目的

提出REACT3D框架，将静态3D场景转换为模拟就绪的交互式副本，使其能够直接用于各种下游任务，并降低关节场景理解大规模研究的门槛。

### 方法

包括四个主要贡献：(i)可打开物体检测和分割，从静态场景中提取候选可移动部分；(ii)关节估计，推断关节类型和运动参数；(iii)隐藏几何形状补全，然后进行交互式物体组装；(iv)交互式场景集成，以广泛支持的格式确保与标准模拟平台的兼容性。

### 主要发现

在各种室内场景的检测/分割和关节指标上实现了最先进的性能，证明了框架的有效性，并为可扩展的交互式场景生成提供了实用基础。

### 结论

REACT3D为可扩展的交互式场景生成提供了实用基础，从而降低了关节场景理解大规模研究的门槛。

### 翻译

交互式3D场景对具身智能日益重要，但由于注释部分分割、运动类型和运动轨迹的劳动密集型过程，现有数据集仍然有限。我们提出了REACT3D，一个可扩展的零样本框架，将静态3D场景转换为具有一致几何形状的模拟就绪交互式副本，可直接用于各种下游任务。我们的贡献包括：(i)可打开物体检测和分割，从静态场景中提取候选可移动部分；(ii)关节估计，推断关节类型和运动参数；(iii)隐藏几何形状补全，然后进行交互式物体组装；(iv)交互式场景集成，以广泛支持的格式确保与标准模拟平台的兼容性。我们在各种室内场景的检测/分割和关节指标上实现了最先进的性能，证明了我们框架的有效性，并为可扩展的交互式场景生成提供了实用基础，从而降低了关节场景理解大规模研究的门槛。我们的项目页面是https://react3d.github.io/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何将静态3D场景转换为交互式物理3D场景的问题。这个问题很重要，因为现有数据集在标注部分分割、运动类型和运动轨迹方面非常耗时，限制了交互式3D场景的发展。而交互式3D场景对虚拟现实、游戏制作和机器人系统训练等应用至关重要，这些应用需要大量既能提供照片级真实感渲染，又能支持物理上合理交互的3D数据集。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到静态3D场景生成已相对成熟，但交互式场景生成仍处于早期阶段。他们提出利用视觉基础模型和视觉语言模型进行零样本转换，无需额外数据收集或计算密集型生成。他们借鉴了现有工作：使用RAM++进行对象识别，Grounded SAM进行分割，OPDMulti进行关节估计，以及类似DRAWER的多视图融合方法，但进行了改进以提高鲁棒性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用视觉基础模型和视觉语言模型从静态3D场景中恢复物体铰接，生成物理启发的3D场景，同时保持原始几何和外观。整体流程包括：1)开放词汇检测和分割，识别可打开物体并提取可移动部分；2)关节估计，确定运动类型和参数；3)隐藏几何生成，完成物体内部空腔；4)交互场景集成，将交互对象与静态背景结合并导出为兼容格式。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)开放词汇检测减轻标签偏差；2)改进的2D到3D分割提高鲁棒性；3)关节细化基于物体几何提高准确性；4)隐藏几何生成使交互更真实；5)广泛的平台兼容性。相比之前工作，REACT3D无需多状态观察或用户交互，不需要仔细分割处理遮挡，比单图像方法提供更一致结果，且在把手缺失情况下表现更好。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'REACT3D提出了一种可扩展的零样本框架，将静态3D场景转换为具有物理交互能力的数字孪生体，为具身智能研究提供了实用基础。'}


### 论文摘要

Interactive 3D scenes are increasingly vital for embodied intelligence, yet existing datasets remain limited due to the labor-intensive process of annotating part segmentation, kinematic types, and motion trajectories. We present REACT3D, a scalable zero-shot framework that converts static 3D scenes into simulation-ready interactive replicas with consistent geometry, enabling direct use in diverse downstream tasks. Our contributions include: (i) openable-object detection and segmentation to extract candidate movable parts from static scenes, (ii) articulation estimation that infers joint types and motion parameters, (iii) hidden-geometry completion followed by interactive object assembly, and (iv) interactive scene integration in widely supported formats to ensure compatibility with standard simulation platforms. We achieve state-of-the-art performance on detection/segmentation and articulation metrics across diverse indoor scenes, demonstrating the effectiveness of our framework and providing a practical foundation for scalable interactive scene generation, thereby lowering the barrier to large-scale research on articulated scene understanding. Our project page is https://react3d.github.io/

---

## 10. IP-Augmented Multi-Modal Malicious URL Detection Via Token-Contrastive Representation Enhancement and Multi-Granularity Fusion

**论文链接:** [http://arxiv.org/abs/2510.12395v1](http://arxiv.org/abs/2510.12395v1)

**作者:** Ye Tian, Yanqiu Yu, Liangliang Song, Zhiquan Liu, Yanbin Wang, Jianguo Sun

**发布时间:** 2025-10-14

### GPT解析

### 总结

本文提出了一种名为CURL-IP的新型多模态恶意URL检测框架，解决了现有方法在处理URL的非自然层次结构、字符级混淆以及整合网络级信号方面的局限性。该框架包含三个关键创新组件，能够同时保留细粒度的词汇线索、上下文语义并整合网络级信号。

### 背景

恶意URL检测仍然是网络安全的关键挑战，因为攻击者越来越多地采用复杂的规避技术，包括混淆、字符级扰动和对抗性攻击。虽然预训练语言模型如BERT在URL分析任务中显示出潜力，但当前实现存在三个主要局限：无法有效建模URL的非自然层次结构、对字符级混淆的敏感性不足，以及缺乏整合辅助网络级信号（如IP地址）的机制。

### 目的

解决当前恶意URL检测方法中的三个主要局限：无法有效建模URL的非自然层次结构、对字符级混淆的敏感性不足，以及缺乏整合网络级信号的机制。提出一个能够同时保留细粒度词汇线索、上下文语义并整合网络级信号的先进检测框架。

### 方法

提出CURL-IP，一个先进的多模态检测框架，包含三个关键创新：(1) Token-Contrastive Representation Enhancer：通过令牌感知对比学习增强子词令牌表示，产生更具区分性和各向同性的嵌入；(2) Cross-Layer Multi-Scale Aggregator：通过卷积操作和门控MLP对Transformer输出进行层次聚合，捕获跨层的局部和全局语义模式；(3) Blockwise Multi-Modal Coupler：将URL-IP特征分解为局部块单元，并在块级别计算跨模态注意力权重，实现细粒度的跨模态交互。

### 主要发现

在大型真实世界数据集上的评估显示，该框架在二元和多类分类任务中显著优于最先进的基线方法。

### 结论

CURL-IP框架通过其创新的架构设计，能够同时处理URL的细粒度特征、上下文语义和网络级信号，有效提高了恶意URL检测的性能。

### 翻译

恶意URL检测仍然是一个关键的网络安全挑战，因为对手越来越多地使用复杂的规避技术，包括混淆、字符级扰动和对抗性攻击。虽然像BERT这样的预训练语言模型在URL分析任务中显示出潜力，但当前实现中仍然存在三个局限性：无法有效建模URL的非自然层次结构，对字符级混淆的敏感性不足，以及缺乏整合辅助网络级信号（如IP地址）的机制——这些都是稳健检测所必需的。为了解决这些挑战，我们提出了CURL-IP，一个先进的多模态检测框架，包含三个关键创新：令牌对比表示增强器，通过令牌感知对比学习增强子词令牌表示；跨层多尺度聚合器，通过卷积操作和门控MLP对Transformer输出进行层次聚合；块级多模态耦合器，将URL-IP特征分解为局部块单元并计算跨模态注意力权重。这种架构能够同时保留细粒度的词汇线索、上下文语义和整合网络级信号。我们在大型真实世界数据集上的评估显示，该框架在二元和多类分类任务中显著优于最先进的基线方法。


### 论文摘要

Malicious URL detection remains a critical cybersecurity challenge as adversaries increasingly employ sophisticated evasion techniques including obfuscation, character-level perturbations, and adversarial attacks. Although pre-trained language models (PLMs) like BERT have shown potential for URL analysis tasks, three limitations persist in current implementations: (1) inability to effectively model the non-natural hierarchical structure of URLs, (2) insufficient sensitivity to character-level obfuscation, and (3) lack of mechanisms to incorporate auxiliary network-level signals such as IP addresses-all essential for robust detection. To address these challenges, we propose CURL-IP, an advanced multi-modal detection framework incorporating three key innovations: (1) Token-Contrastive Representation Enhancer, which enhances subword token representations through token-aware contrastive learning to produce more discriminative and isotropic embeddings; (2) Cross-Layer Multi-Scale Aggregator, employing hierarchical aggregation of Transformer outputs via convolutional operations and gated MLPs to capture both local and global semantic patterns across layers; and (3) Blockwise Multi-Modal Coupler that decomposes URL-IP features into localized block units and computes cross-modal attention weights at the block level, enabling fine-grained inter-modal interaction. This architecture enables simultaneous preservation of fine-grained lexical cues, contextual semantics, and integration of network-level signals. Our evaluation on large-scale real-world datasets shows the framework significantly outperforms state-of-the-art baselines across binary and multi-class classification tasks.

---

## 11. Can Representation Gaps Be the Key to Enhancing Robustness in Graph-Text Alignment?

**论文链接:** [http://arxiv.org/abs/2510.12087v1](http://arxiv.org/abs/2510.12087v1)

**作者:** Heng Zhang, Tianyi Zhang, Yuling Shi, Xiaodong Gu, Yaomin Shen, Zijian Zhang, Yilei Yuan, Hao Zhang, Jin Huang

**发布时间:** 2025-10-14

### GPT解析

### 总结

文本属性图(TAGs)上的表示学习结合了结构连接与丰富的文本语义，但在当前方法中，过度对齐会导致性能下降。作者提出了LLM4GTA框架，通过保留表示间隙来维持模态特定知识并提高迁移性能。

### 背景

当前文本属性图表示学习方法主要依赖对比学习来最大化跨模态相似性，假设图和文本表示之间的更紧密耦合可以提高迁移性能。

### 目的

解决现有方法中过度对齐导致的性能下降问题，提出一种保留表示间隙的框架，以维持模态特定知识并提高迁移性能。

### 方法

提出LLM4GTA框架，包含自适应间隙保留模块防止过度对齐，以及模内补偿机制使用辅助分类器增强图空间的判别能力。

### 主要发现

经验分析表明，无论是自然间隙扩大还是强制间隙减小都会导致性能下降，这是因为图编码器捕获拓扑模式而文本编码器捕获语义结构，两者之间存在几何不兼容性。

### 结论

保留表示间隙作为维持模态特定知识的几何必要性可以改善零样本和少样本场景下的性能表现。

### 翻译

文本属性图(TAGs)上的表示学习将结构连接与丰富的文本语义相结合，使能够在多个领域应用。当前方法主要依赖对比学习来最大化跨模态相似性，假设图和文本表示之间的更紧密耦合可以提高迁移性能。然而，我们的经验分析显示，无论是自然间隙扩大还是强制间隙减小都会通过破坏预训练知识结构和损害泛化能力而导致性能下降。这是由于编码器之间的几何不兼容性造成的，其中图编码器捕获拓扑模式，而文本编码器捕获语义结构。过度对齐将这些不同的空间压缩到共享子空间中，导致结构崩溃，同时削弱了拓扑推理和语义理解。我们提出LLM4GTA，一个间隙感知的对齐框架，将表示间隙保留为维持模态特定知识和提高迁移性能的几何必要性。LLM4GTA包括自适应间隙保留模块，通过监控相似性演变防止过度对齐，以及模内补偿机制，使用图空间中的辅助分类器增强判别能力。大量实验表明，在零样本和少样本场景下，与现有方法相比显示出显著改进。


### 论文摘要

Representation learning on text-attributed graphs (TAGs) integrates structural connectivity with rich textual semantics, enabling applications in diverse domains. Current methods largely rely on contrastive learning to maximize cross-modal similarity, assuming tighter coupling between graph and text representations improves transfer performance. However, our empirical analysis reveals that both natural gap expansion and forced gap reduction result in performance degradation by disrupting pre-trained knowledge structures and impairing generalization. This arises from the geometric incompatibility between encoders, where graph encoders capture topological patterns, while text encoders capture semantic structures. Over-alignment compresses these distinct spaces into shared subspaces, causing structure collapse that diminishes both topological reasoning and semantic understanding. We propose \textbf{LLM4GTA}, a gap-aware alignment framework that preserves representation gaps as geometric necessities for maintaining modality-specific knowledge and improving transfer performance. LLM4GTA includes an adaptive gap preservation module to prevent over-alignment by monitoring similarity evolution and an intra-modal compensation mechanism that boosts discriminative power using auxiliary classifiers in graph space. Extensive experiments show significant improvements over existing methods in zero-shot and few-shot scenarios.

---

## 12. GraphShaper: Geometry-aware Alignment for Improving Transfer Learning in Text-Attributed Graphs

**论文链接:** [http://arxiv.org/abs/2510.12085v1](http://arxiv.org/abs/2510.12085v1)

**作者:** Heng Zhang, Tianyi Zhang, Yuling Shi, Xiaodong Gu, Yaomin Shen, Haochen You, Zijian Zhang, Yilei Yuan, Jin Huang

**发布时间:** 2025-10-14

### GPT解析

### 总结

论文提出GraphShaper几何感知框架，通过多几何专业化解决图基础模型在结构边界处性能下降问题，实现零样本设置下在引用网络和社会网络上分别提高9.47%和7.63%的准确率。

### 背景

图基础模型是一种在不同图域中学习可迁移表示的变革性范式。最近方法利用大型语言模型通过对比学习将图和文本模态统一到共享表示空间，但系统评估显示在结构边界处性能显著下降，准确率损失超过20个百分点。

### 目的

设计一个能够尊重图结构内在几何多样性的对齐框架，解决当前图基础模型在结构边界处性能下降的问题。

### 方法

提出GraphShaper框架，采用针对不同几何空间的专业网络，动态计算融合权重，基于局部结构特征自适应地集成几何特性，在对齐文本嵌入前保持结构完整性。

### 主要发现

当前方法假设所有图结构可在单一欧几里得空间编码，但实际上树结构需要双曲几何保持分层分支，循环模式依赖球面几何的闭合性质。在结构边界处，节点经历冲突的几何约束，统一编码空间无法解决。

### 结论

GraphShaper通过多几何专业化和自适应融合策略，有效解决了图基础模型在结构边界处的性能下降问题，显著提高了零样本设置下的准确率。

### 翻译

图基础模型代表了一种在不同图域中学习可迁移表示的变革性范式。最近的方法利用大型语言模型通过对比学习将图和文本模态统一到共享表示空间中。然而，系统评估显示，在具有不同拓扑模式汇聚的结构边界处，性能显著下降，准确率损失超过20个百分点。这个问题源于一个关键限制：当前方法假设所有图结构都可以在单一欧几里得空间内编码。实际上，树结构需要双曲几何来保持分层分支，而循环模式则依赖于球面几何的闭合性质。在结构边界处，节点经历冲突的几何约束，统一的编码空间无法解决。这提出了一个关键挑战：能否设计尊重图结构内在几何多样性的对齐框架？我们介绍了GraphShaper，一个几何感知框架，通过多几何专业化增强图编码。我们的方法采用针对不同几何空间的专业网络，动态计算融合权重，基于局部结构特征自适应地集成几何特性。这种自适应融合在对齐文本嵌入之前保持结构完整性。大量实验证明，GraphShaper在零样本设置下在引用网络上实现了9.47%的准确率提升，在社会网络上实现了7.63%的准确率提升。


### 论文摘要

Graph foundation models represent a transformative paradigm for learning transferable representations across diverse graph domains. Recent methods leverage large language models to unify graph and text modalities into a shared representation space using contrastive learning. However, systematic evaluations reveal significant performance degradation at structural boundaries where distinct topological patterns converge, with accuracy losses exceeding 20 percentage points. This issue arises from a key limitation: current methods assume all graph structures can be encoded within a single Euclidean space. In reality, tree structures require hyperbolic geometry to preserve hierarchical branching, while cyclic patterns depend on spherical geometry for closure properties. At structural boundaries, nodes experience conflicting geometric constraints that uniform encoding spaces cannot resolve. This raises a crucial challenge: \textbf{Can alignment frameworks be designed to respect the intrinsic geometric diversity of graph structures?} We introduce \textbf{GraphShaper}, a geometry-aware framework that enhances graph encoding through multi-geometric specialization. Our approach employs expert networks tailored to different geometric spaces, dynamically computing fusion weights to adaptively integrate geometric properties based on local structural characteristics. This adaptive fusion preserves structural integrity before alignment with text embeddings. Extensive experiments demonstrate that GraphShaper achieves 9.47\% accuracy improvements on citation networks and 7.63\% on social networks in zero-shot settings.

---

## 13. MEASURE: Multi-scale Minimal Sufficient Representation Learning for Domain Generalization in Sleep Staging

**论文链接:** [http://arxiv.org/abs/2510.12070v1](http://arxiv.org/abs/2510.12070v1)

**作者:** Sangmin Jo, Jee Seok Yoon, Wootaek Jeong, Kwanseok Oh, Heung-Il Suk

**发布时间:** 2025-10-14

**备注:** 12 page, 7 figures, uses IEEE.sty

### GPT解析

### 总结

本文提出了一种名为MEASURE的新型深度学习框架，用于解决睡眠分期模型在分布外场景中的泛化问题。该框架通过减少领域相关信息同时保留重要的时频特征，显著提升了模型在未见过的数据上的性能。

### 背景

基于深度学习的自动睡眠分期技术近年来性能显著提升，对睡眠障碍诊断至关重要。然而，这些模型在处理不同受试者的生理信号时存在泛化困难，导致在分布外场景中性能下降。领域泛化方法，特别是对比学习，被研究用于解决这一问题，但现有方法往往无法充分提取领域不变特征。

### 目的

开发一种能够有效减少领域相关信息同时保留睡眠分期所需关键时频特征的新型框架，以提高模型在未见数据上的泛化能力。

### 方法

提出了MEASURE（Multi-scale Minimal Sufficient Representation Learning）框架，该框架采用多尺度方法最小化充分表示学习，能够在减少领域相关信息的同时，保留多级特征中编码的多样时频信息。

### 主要发现

在SleepEDF-20和MASS等公开睡眠分期基准数据集上的详尽实验表明，所提出的MEASURE方法持续优于当前最先进的方法，证明了其在处理领域差异方面的有效性。

### 结论

MEASURE框架通过针对性地减少过度领域相关信息同时保留关键特征，成功弥合了领域差距，显著提高了睡眠分期模型在分布外场景中的泛化性能。

### 翻译

基于深度学习的自动睡眠分期在性能上已取得显著进展，并在睡眠障碍诊断中起着关键作用。然而，由于生理信号的变异性，这些模型往往难以在未见过的受试者上泛化，导致在分布外场景中性能下降。为了解决这个问题，最近研究了领域泛化方法，以确保在训练期间对未见领域有泛化性能。在这些技术中，对比学习已证明通过在不同领域间对齐同类样本来学习领域不变特征的有效性。尽管有潜力，但许多现有方法不足以提取充分的领域不变表示，因为它们没有明确解决样本间非共享信息中嵌入的领域特征。在本文中，我们认为减轻这种领域相关属性（称为过度领域相关信息）是弥合领域差距的关键。然而，直接减轻领域相关属性的策略往往对高级信息特征过拟合，限制了利用多级特征中编码的多样时频信息的能力。为了解决这些局限性，我们提出了一个新颖的MEASURE（多尺度最小充分表示学习）框架，该框架在减少领域相关信息的同时，有效保留了睡眠分期分类的基本时频特征。在公开可用的睡眠分期基准数据集SleepEDF-20和MASS上进行的详尽实验中，我们提出的方法持续优于最先进的方法。我们的代码可在 https://github.com/ku-milab/Measure 获取。


### 论文摘要

Deep learning-based automatic sleep staging has significantly advanced in performance and plays a crucial role in the diagnosis of sleep disorders. However, those models often struggle to generalize on unseen subjects due to variability in physiological signals, resulting in degraded performance in out-of-distribution scenarios. To address this issue, domain generalization approaches have recently been studied to ensure generalized performance on unseen domains during training. Among those techniques, contrastive learning has proven its validity in learning domain-invariant features by aligning samples of the same class across different domains. Despite its potential, many existing methods are insufficient to extract adequately domain-invariant representations, as they do not explicitly address domain characteristics embedded within the unshared information across samples. In this paper, we posit that mitigating such domain-relevant attributes-referred to as excess domain-relevant information-is key to bridging the domain gap. However, the direct strategy to mitigate the domain-relevant attributes often overfits features at the high-level information, limiting their ability to leverage the diverse temporal and spectral information encoded in the multiple feature levels. To address these limitations, we propose a novel MEASURE (Multi-scalE minimAl SUfficient Representation lEarning) framework, which effectively reduces domain-relevant information while preserving essential temporal and spectral features for sleep stage classification. In our exhaustive experiments on publicly available sleep staging benchmark datasets, SleepEDF-20 and MASS, our proposed method consistently outperformed state-of-the-art methods. Our code is available at : https://github.com/ku-milab/Measure

---

## 14. MammoDINO: Anatomically Aware Self-Supervision for Mammographic Images

**论文链接:** [http://arxiv.org/abs/2510.11883v1](http://arxiv.org/abs/2510.11883v1)

**作者:** Sicheng Zhou, Lei Wu, Cao Xiao, Parminder Bhatia, Taha Kass-Hout

**发布时间:** 2025-10-13

**备注:** 5 pages

### GPT解析

### 总结

MammoDINO是一种专门为乳腺X线摄影设计的自监督学习框架，通过在大规模数据上预训练，实现了在乳腺癌筛查任务上的最先进性能，并具有良好的泛化能力，为计算机辅助诊断提供了无标注的基础。

### 背景

自监督学习已在一般领域的视觉编码器训练中取得变革性进展，但在医学影像领域应用不足，主要原因是数据有限和领域特定偏差。

### 目的

开发一种适用于乳腺X线摄影的自监督学习框架，捕捉临床上有意义的特征，提高乳腺癌筛查的诊断效率，减轻放射科医生的工作量。

### 方法

构建MammoDINO框架，在140万张乳腺X线图像上预训练；引入乳腺组织感知的数据增强采样器，用于图像级和补丁级监督；设计跨切片对比学习目标，利用3D数字乳腺断层合成结构进行2D预训练。

### 主要发现

MammoDINO在多个乳腺癌筛查任务上取得了最先进的性能，并在五个基准数据集上表现出良好的泛化能力。

### 结论

MammoDINO为乳腺X线摄影的多用途计算机辅助诊断工具提供了一种可扩展的无标注基础，有助于减轻放射科医生的工作量，提高乳腺癌筛查的诊断效率。

### 翻译

自监督学习(SSL)已经在一般领域的视觉编码器训练中取得变革性进展，但由于数据有限和领域特定偏差，在医学影像中的应用仍然不足。我们提出了MammoDINO，这是一种用于乳腺X线摄影的新型SSL框架，在140万张乳腺X线图像上进行了预训练。为了捕捉临床上有意义的特征，我们引入了一种乳腺组织感知的数据增强采样器，用于图像级和补丁级监督，以及跨切片对比学习目标，利用3D数字乳腺断层合成(DBT)结构进行2D预训练。MammoDINO在多个乳腺癌筛查任务上取得了最先进的性能，并在五个基准数据集上表现出良好的泛化能力。它为乳腺X线摄影的多用途计算机辅助诊断(CAD)工具提供了一种可扩展的无标注基础，有助于减轻放射科医生的工作量，提高乳腺癌筛查的诊断效率。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自监督学习在乳腺X光摄影(mammography)领域的应用不足问题。这个问题很重要，因为乳腺癌是美国女性最常见的癌症，乳腺X光是主要筛查方式，但准确解读具有挑战性，且现有计算机辅助诊断工具依赖大量标注数据，而标注的乳腺X光数据有限，限制了模型效果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有自监督学习框架在乳腺X光图像中的局限性：随机裁剪可能过度采样不相关背景，传统对比学习仅限单个2D切片无法捕捉3D DBT数据的跨切片结构连贯性。作者借鉴了DINOv2框架，参考了RAD-DINO和MedCoSS等医学影像自监督学习方法，并针对乳腺X光特点进行了创新设计。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是：1)确保模型专注于临床有意义的乳腺组织区域而非背景；2)利用3D DBT的跨切片结构连贯性。整体流程：收集140万乳腺X光图像→预处理→基于ViT架构的模型预训练→结合乳腺组织感知的DINO损失、iBOT损失和3D DBT相邻切片损失→在五个基准数据集上微调和评估多种乳腺癌筛查任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)乳腺组织感知的数据增强采样器，确保模型关注临床相关区域；2)3D DBT相邻切片损失，捕捉跨切片解剖连续性。不同之处：相比通用SSL框架(如DINOv2)，避免了过度采样背景和忽略3D结构；相比放射学定制SSL(如RadDINO)，专门针对乳腺解剖结构优化；相比文本引导方法(如BiomedCLIP)，无需文本监督仍能取得优越性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MammoDINO通过引入乳腺组织感知的数据增强和3D跨切片学习机制，解决了自监督学习在乳腺X光图像中的局限性，实现了乳腺癌筛查任务的先进性能，为开发更有效的计算机辅助诊断系统提供了无需标注的基础模型。'}


### 论文摘要

Self-supervised learning (SSL) has transformed vision encoder training in general domains but remains underutilized in medical imaging due to limited data and domain specific biases. We present MammoDINO, a novel SSL framework for mammography, pretrained on 1.4 million mammographic images. To capture clinically meaningful features, we introduce a breast tissue aware data augmentation sampler for both image-level and patch-level supervision and a cross-slice contrastive learning objective that leverages 3D digital breast tomosynthesis (DBT) structure into 2D pretraining. MammoDINO achieves state-of-the-art performance on multiple breast cancer screening tasks and generalizes well across five benchmark datasets. It offers a scalable, annotation-free foundation for multipurpose computer-aided diagnosis (CAD) tools for mammogram, helping reduce radiologists' workload and improve diagnostic efficiency in breast cancer screening.

---

## 15. Improving Knowledge Graph Embeddings through Contrastive Learning with Negative Statements

**论文链接:** [http://arxiv.org/abs/2510.11868v1](http://arxiv.org/abs/2510.11868v1)

**作者:** Rita T. Sousa, Heiko Paulheim

**发布时间:** 2025-10-13

**备注:** Accepted at the Thirteenth International Conference on Knowledge  Capture (K-CAP 2025)

### GPT解析

### 总结

本文提出了一种新的知识图谱嵌入方法，通过整合明确的否定陈述来改进知识嵌入学习过程，在链接预测和三元组分类任务上取得了优于现有模型的性能。

### 背景

知识图谱以结构化三元组表示信息，是问答、链接预测和推荐系统等多种应用的基础。现有图嵌入方法大多依赖封闭世界假设，将缺失三元组视为错误，这与现实世界知识图谱的开放世界假设相矛盾，且很少考虑明确的否定陈述。

### 目的

开发一种新方法，将明确声明的否定陈述整合到知识嵌入学习过程中，以改进知识图谱的表示和预测能力。

### 方法

采用双模型架构，两个嵌入模型并行训练：一个在正陈述上训练，另一个在负陈述上训练。训练过程中，每个模型通过破坏正样本生成负样本，并使用另一个模型的评分选择最可能的候选样本。

### 主要发现

在通用和特定领域知识图谱上的大量实验表明，该方法在链接预测和三元组分类任务上优于最先进的嵌入模型，证明了整合有意义的负知识到嵌入学习中的价值。

### 结论

通过整合明确的否定陈述，该方法有效提高了知识图谱嵌入的预测性能，为处理开放世界假设下的知识图谱提供了新思路。

### 翻译

知识图谱将信息表示为结构化三元组，并作为问答、链接预测和推荐系统等广泛应用的基础。探索知识图谱的一个主要研究方向是图嵌入方法，其中实体和关系在低维向量空间中表示，以捕获底层语义和结构。然而，大多数现有方法依赖于封闭世界假设或局部封闭世界假设等假设，将缺失的三元组视为错误。这与许多现实世界知识图谱所基于的开放世界假设形成对比。此外，虽然明确陈述的否定陈述有助于区分错误和未知的三元组，但它们很少被纳入知识图谱，在嵌入训练过程中也经常被忽视。在这项工作中，我们介绍了一种新方法，将明确声明的否定陈述整合到知识嵌入学习过程中。我们的方法采用双模型架构，两个嵌入模型并行训练，一个在正陈述上训练，另一个在负陈述上训练。在训练过程中，每个模型通过破坏正样本生成负样本，并使用另一个模型的评分选择最可能的候选样本。所提出的方法在通用和特定领域知识图谱上进行了评估，重点关注链接预测和三元组分类任务。大量实验表明，我们的方法优于最先进的嵌入模型，证明了将有意义的负知识整合到嵌入学习中的价值。


### 论文摘要

Knowledge graphs represent information as structured triples and serve as the backbone for a wide range of applications, including question answering, link prediction, and recommendation systems. A prominent line of research for exploring knowledge graphs involves graph embedding methods, where entities and relations are represented in low-dimensional vector spaces that capture underlying semantics and structure. However, most existing methods rely on assumptions such as the Closed World Assumption or Local Closed World Assumption, treating missing triples as false. This contrasts with the Open World Assumption underlying many real-world knowledge graphs. Furthermore, while explicitly stated negative statements can help distinguish between false and unknown triples, they are rarely included in knowledge graphs and are often overlooked during embedding training.   In this work, we introduce a novel approach that integrates explicitly declared negative statements into the knowledge embedding learning process. Our approach employs a dual-model architecture, where two embedding models are trained in parallel, one on positive statements and the other on negative statements. During training, each model generates negative samples by corrupting positive samples and selecting the most likely candidates as scored by the other model. The proposed approach is evaluated on both general-purpose and domain-specific knowledge graphs, with a focus on link prediction and triple classification tasks. The extensive experiments demonstrate that our approach improves predictive performance over state-of-the-art embedding models, demonstrating the value of integrating meaningful negative knowledge into embedding learning.

---

## 16. Combining Euclidean and Hyperbolic Representations for Node-level Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2510.11827v1](http://arxiv.org/abs/2510.11827v1)

**作者:** Simone Mungari, Ettore Ritacco, Pietro Sabatino

**发布时间:** 2025-10-13

### GPT解析

### 总结

Janus框架通过联合利用欧几里得和双曲图神经网络，有效解决了节点级异常检测的挑战，通过对比学习对齐不同空间中的嵌入，识别难以协调的节点视图来检测异常。

### 背景

节点级异常检测(NAD)具有挑战性，因为存在多样的结构模式和特征分布。NAD是一个关键任务，应用于欺诈检测、网络安全、推荐系统等多个领域。

### 目的

提出一个名为Janus的框架，用于有效解决节点级异常检测问题。

### 方法

每个节点由两个视图描述：原始特征和从随机游走和度派生的结构特征，然后嵌入到欧几里得和双曲空间中。使用配备对比学习目标作为正则化项的多图自编码器框架，对齐欧几里得和双曲空间中的嵌入，突出显示那些视图难以协调的节点，这些节点可能是异常的。

### 主要发现

在四个真实数据集上的实验表明，Janus始终优于浅层和深度基线方法， empirically demonstrating that combining multiple geometric representations provides a robust and effective approach for identifying subtle and complex anomalies in graphs.

### 结论

组合多种几何表示是一种稳健有效的方法，用于识别图中的微妙和复杂异常。

### 翻译

节点级异常检测(NAD)具有挑战性，因为存在多样的结构模式和特征分布。因此，NAD是一个关键任务，应用于从欺诈检测、网络安全到推荐系统的多个领域。我们引入了Janus，一个联合利用欧几里得和双曲图神经网络来捕捉节点表示互补方面的框架。每个节点由两个视图描述，由原始特征和从随机游走和度派生的结构特征组成，然后嵌入到欧几里得和双曲空间中。配备对比学习目标作为正则化项的多图自编码器框架，对齐欧几里得和双曲空间中的嵌入，突出显示那些视图难以协调的节点，因此可能是异常的。在四个真实数据集上的实验表明，Janus始终优于浅层和深度基线方法， empirically demonstrating that combining multiple geometric representations provides a robust and effective approach for identifying subtle and complex anomalies in graphs.


### 论文摘要

Node-level anomaly detection (NAD) is challenging due to diverse structural patterns and feature distributions. As such, NAD is a critical task with several applications which range from fraud detection, cybersecurity, to recommendation systems. We introduce Janus, a framework that jointly leverages Euclidean and Hyperbolic Graph Neural Networks to capture complementary aspects of node representations. Each node is described by two views, composed by the original features and structural features derived from random walks and degrees, then embedded into Euclidean and Hyperbolic spaces. A multi Graph-Autoencoder framework, equipped with a contrastive learning objective as regularization term, aligns the embeddings across the Euclidean and Hyperbolic spaces, highlighting nodes whose views are difficult to reconcile and are thus likely anomalous. Experiments on four real-world datasets show that Janus consistently outperforms shallow and deep baselines, empirically demonstrating that combining multiple geometric representations provides a robust and effective approach for identifying subtle and complex anomalies in graphs.

---

## 17. Personalized Federated Fine-Tuning of Vision Foundation Models for Healthcare

**论文链接:** [http://arxiv.org/abs/2510.12741v1](http://arxiv.org/abs/2510.12741v1)

**作者:** Adam Tupper, Christian Gagné

**发布时间:** 2025-10-14

**备注:** Accepted to the Symposium on Model Accountability, Sustainability and  Healthcare (SMASH) 2025

### GPT解析

### 总结

该研究提出了一种新的个性化联邦微调方法，通过学习正交LoRA适配器来解耦通用知识和客户特定知识，使医疗领域的每个参与方能够充分利用自己的数据和他人的数据，解决了医疗数据隐私保护与模型性能之间的矛盾。

### 背景

基础模型为AI在医疗领域的应用开辟了新可能性，但即使在健康数据上预训练，仍需针对特定下游任务微调。尽管基础模型减少了训练数据需求，但获取足够数据仍具挑战性，部分原因是医疗数据共享和聚合受到患者隐私保护限制。

### 目的

开发一种能够在保护患者隐私的前提下，有效利用多方医疗数据的联邦微调方法，使各参与机构能够充分利用自有数据和他人的数据来提高模型性能。

### 方法

提出一种新的个性化联邦微调方法，学习正交LoRA适配器来解耦通用知识和客户特定知识，使每个客户端能够同时利用自有数据和来自其他参与方的数据。

### 主要发现

在实际联邦医学成像任务上的初步结果表明，该方法与当前联邦微调方法具有竞争力，能够有效平衡数据隐私保护与模型性能提升。

### 结论

通过联邦学习框架下的正交LoRA适配器方法，解决了医疗AI中数据隐私保护与模型性能之间的矛盾，为医疗领域的基础模型应用提供了有效解决方案。

### 翻译

基础模型为AI在医疗领域的应用开辟了新的可能性。然而，即使在健康数据上预训练，它们仍需要针对特定的下游任务进行微调。此外，尽管基础模型减少了实现良好性能所需的训练数据量，但获取足够的数据仍然是一个挑战。这部分是由于为了保护患者隐私，限制了不同来源数据的共享和聚合。一个可能的解决方案是通过多个参与方（即医院、诊所等）之间的联邦学习来微调基础模型。在这项工作中，我们提出了一种新的个性化联邦微调方法，学习正交LoRA适配器来解耦通用知识和客户特定知识，使每个客户能够充分利用自己的数据和他人数据。我们在实际联邦医学成像任务上的初步结果表明，我们的方法与当前的联邦微调方法具有竞争力。


### 论文摘要

Foundation models open up new possibilities for the use of AI in healthcare. However, even when pre-trained on health data, they still need to be fine-tuned for specific downstream tasks. Furthermore, although foundation models reduce the amount of training data required to achieve good performance, obtaining sufficient data is still a challenge. This is due, in part, to restrictions on sharing and aggregating data from different sources to protect patients' privacy. One possible solution to this is to fine-tune foundation models via federated learning across multiple participating clients (i.e., hospitals, clinics, etc.). In this work, we propose a new personalized federated fine-tuning method that learns orthogonal LoRA adapters to disentangle general and client-specific knowledge, enabling each client to fully exploit both their own data and the data of others. Our preliminary results on real-world federated medical imaging tasks demonstrate that our approach is competitive against current federated fine-tuning methods.

---

## 18. T(R,O) Grasp: Efficient Graph Diffusion of Robot-Object Spatial Transformation for Cross-Embodiment Dexterous Grasping

**论文链接:** [http://arxiv.org/abs/2510.12724v1](http://arxiv.org/abs/2510.12724v1)

**作者:** Xin Fei, Zhixuan Xu, Huaicong Fang, Tianrui Zhang, Lin Shao

**发布时间:** 2025-10-14

**备注:** 12 pages, 14 figures

### GPT解析

### 总结

T(R,O)Grasp是一种基于扩散的框架，能够高效生成准确和多样化的抓取动作，适用于多种机器人手，在实验中取得了94.83%的平均成功率，推理速度为0.21秒，每秒可处理41个抓取动作，显著优于现有方法。

### 背景

灵巧抓取在机器人学中仍然是一个核心挑战，这主要由于其高维状态和动作空间的复杂性。

### 目的

引入T(R,O)Grasp，一种基于扩散的框架，用于高效生成准确和多样化的抓取动作，适用于多种机器人手。

### 方法

核心是T(R,O)图，一种统一表示方法，建模机器人手和物体间的空间变换并编码其几何属性；结合图扩散模型和高效的逆运动学求解器，支持无条件和有条件的抓取合成。

### 主要发现

在多种灵巧手上进行的实验显示，T(R,O)Grasp平均成功率达94.83%，推理速度0.21秒，在NVIDIA A100 40GB GPU上每秒处理41个抓取动作；该方法在不同实现上具有鲁棒性和泛化能力，显著减少内存消耗，高推理速度使闭环灵巧操作成为可能。

### 结论

T(R,O)Grasp有潜力扩展为灵巧抓取的基础模型。

### 翻译

灵巧抓取由于高维状态和动作空间的复杂性，在机器人学中仍然是一个核心挑战。我们引入了T(R,O)Grasp，一种基于扩散的框架，能够高效生成准确和多样化的抓取动作，适用于多种机器人手。其核心是T(R,O)图，一种统一表示方法，建模机器人手和物体间的空间变换并编码其几何属性。结合图扩散模型和高效的逆运动学求解器，支持无条件和有条件的抓取合成。在多种灵巧手上的广泛实验表明，T(R,O)Grasp在NVIDIA A100 40GB GPU上达到94.83%的平均成功率、0.21秒的推理速度和每秒41个抓取的吞吐量，显著优于现有基线方法。此外，我们的方法在不同实现上具有鲁棒性和泛化能力，同时显著减少内存消耗。更重要的是，高推理速度使闭环灵巧操作成为可能，突显了T(R,O)Grasp扩展为灵巧抓取基础模型的潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何高效生成精确且多样化的灵巧抓取动作（dexterous grasping）问题，特别是在跨机器人平台（cross-embodiment）情况下。这个问题很重要，因为灵巧抓取是实现人类级精确操作的基础能力，对于机器人完成日常任务至关重要；现有方法要么计算效率低下，要么难以在不同机器人平台间泛化；高效的抓取生成对于实时应用和闭环控制至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性（机器人中心方法泛化能力差、物体中心方法计算成本高、交互中心方法如D(R,O)内存消耗大且依赖初始状态）设计了新方法。作者借鉴了扩散模型（DDPM/DDIM）、图神经网络、空间变换表示（SE(3)）和逆运动学求解等现有技术，但创新性地提出了T(R,O) Graph表示和图扩散模型，解决了效率和泛化问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是T(R,O) Graph表示和图扩散模型。T(R,O) Graph将物体和机器人手表示为图中的节点，它们之间的空间变换表示为边；图扩散模型基于此表示生成抓取。整体流程：1) 构建T(R,O) Graph（物体节点和手节点及其空间变换）；2) 图扩散模型前向过程添加噪声；3) 去噪模型预测噪声；4) 反向过程恢复干净抓取；5) 使用Pyroki进行逆运动学求解得到关节值。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) T(R,O) Graph统一表示，显著减少内存使用；2) 高效图扩散模型，支持非条件和条件生成；3) 不依赖初始状态，避免D(R,O)的局限性；4) 高效训练推理（内存减少57%，速度提高3倍）；5) 强大的跨平台泛化能力。相比D(R,O)，成功率更高（94.83% vs 87.53%），内存更少，速度更快，不依赖初始状态。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'T(R,O) Grasp通过创新的图扩散模型和高效的T(R,O) Graph表示，实现了跨平台灵巧抓取的高效生成，在保持高成功率的同时显著提高了推理速度并降低了内存消耗，为实时闭环抓取提供了新解决方案。'}


### 论文摘要

Dexterous grasping remains a central challenge in robotics due to the complexity of its high-dimensional state and action space. We introduce T(R,O) Grasp, a diffusion-based framework that efficiently generates accurate and diverse grasps across multiple robotic hands. At its core is the T(R,O) Graph, a unified representation that models spatial transformations between robotic hands and objects while encoding their geometric properties. A graph diffusion model, coupled with an efficient inverse kinematics solver, supports both unconditioned and conditioned grasp synthesis. Extensive experiments on a diverse set of dexterous hands show that T(R,O) Grasp achieves average success rate of 94.83%, inference speed of 0.21s, and throughput of 41 grasps per second on an NVIDIA A100 40GB GPU, substantially outperforming existing baselines. In addition, our approach is robust and generalizable across embodiments while significantly reducing memory consumption. More importantly, the high inference speed enables closed-loop dexterous manipulation, underscoring the potential of T(R,O) Grasp to scale into a foundation model for dexterous grasping.

---

## 19. Multitask finetuning and acceleration of chemical pretrained models for small molecule drug property prediction

**论文链接:** [http://arxiv.org/abs/2510.12719v1](http://arxiv.org/abs/2510.12719v1)

**作者:** Matthew Adrian, Yunsie Chung, Kevin Boyd, Saee Paliwal, Srimukh Prasad Veccham, Alan C. Cheng

**发布时间:** 2025-10-14

### GPT解析

### 总结

该研究探讨了化学预训练模型在药物发现中的应用，特别是在多任务学习框架下的微调效果。

### 背景

化学预训练模型（基础模型）在药物发现领域受到广泛关注。从自监督训练中提取的一般化学知识有望提高对关键药物发现终点的预测，包括靶点效力和ADMET性质。多任务学习已被成功用于改进预测模型。

### 目的

研究在化学预训练图神经网络模型（如KERMT和KGPT）的微调过程中启用多任务学习的效果。

### 方法

通过比较多任务微调与非预训练图神经网络模型的性能差异，评估了Kinetic GROVER Multi-Task (KERMT)和Knowledge-guided Pre-training of Graph Transformer (KGPT)模型的表现。

### 主要发现

1. 多任务微调显著提高了预训练图神经网络模型的性能；2. 数据量越大，多任务微调KERMT带来的性能提升越显著；3. 发布了两个多任务ADMET数据分割，便于更准确地基准测试多任务深度学习方法；4. 在GitHub上提供了KERMT模型的加速实现，支持工业药物发现工作流程。

### 结论

多任务微调能显著提升化学预训练图神经网络模型在药物发现应用中的性能，特别是在大数据集上，并提供了相关工具和数据资源促进该领域的研究。

### 翻译

化学预训练模型，有时被称为基础模型，正在药物发现应用中引起相当大的关注。从自监督训练中提取的一般化学知识有可能提高对关键药物发现终点的预测，包括靶点效力和ADMET性质。多任务学习已被成功用于改进预测模型。在这里，我们表明，在化学预训练图神经网络模型的微调中启用多任务，如Kinetic GROVER Multi-Task (KERMT)（GROVER模型的增强版本）和Knowledge-guided Pre-training of Graph Transformer (KGPT)，显著优于非预训练的图神经网络模型。令人惊讶的是，我们发现以多任务方式微调KERMT带来的性能提升在数据量较大时最为显著。此外，我们发布了两个多任务ADMET数据分割，以便更准确地基准测试多任务深度学习方法用于药物性质预测。最后，我们在GitHub上提供了KERMT模型的加速实现，使工业药物发现工作流程中的大规模预训练、微调和推理成为可能。


### 论文摘要

Chemical pretrained models, sometimes referred to as foundation models, are receiving considerable interest for drug discovery applications. The general chemical knowledge extracted from self-supervised training has the potential to improve predictions for critical drug discovery endpoints, including on-target potency and ADMET properties. Multi-task learning has previously been successfully leveraged to improve predictive models. Here, we show that enabling multitasking in finetuning of chemical pretrained graph neural network models such as Kinetic GROVER Multi-Task (KERMT), an enhanced version of the GROVER model, and Knowledge-guided Pre-training of Graph Transformer (KGPT) significantly improves performance over non-pretrained graph neural network models. Surprisingly, we find that the performance improvement from finetuning KERMT in a multitask manner is most significant at larger data sizes. Additionally, we publish two multitask ADMET data splits to enable more accurate benchmarking of multitask deep learning methods for drug property prediction. Finally, we provide an accelerated implementation of the KERMT model on GitHub, unlocking large-scale pretraining, finetuning, and inference in industrial drug discovery workflows.

---

## 20. SAIL-Embedding Technical Report: Omni-modal Embedding Foundation Model

**论文链接:** [http://arxiv.org/abs/2510.12709v1](http://arxiv.org/abs/2510.12709v1)

**作者:** Lin Lin, Jiefeng Long, Zhihe Wan, Yuchi Wang, Dingkang Yang, Shuang Yang, Yueyang Yao, Xu Chen, Zirui Guo, Shengqiang Li, Weiran Li, Hanyu Li, Yaling Mou, Yan Qiu, Haiyang Yu, Xiao Liang, Hongsheng Li, Chao Feng

**发布时间:** 2025-10-14

**备注:** Technical Report

### GPT解析

### 总结

SAIL-Embedding是一种全模态嵌入基础模型，通过定制的训练策略和架构设计解决了现有多模态嵌入模型在实际应用中面临的挑战，并在各种检索任务和实际业务场景中取得了优异性能。

### 背景

多模态嵌入模型旨在产生信息丰富的统一表示以支持各种跨模态任务。尽管从基于CLIP的双塔架构到大型视觉语言模型的发展有前景，但现有模型在实际应用和业务场景中仍面临模态支持有限、训练机制不稳定和工业领域差距等挑战。

### 目的

开发一种能够解决现有多模态嵌入模型在实际应用中面临挑战的全模态嵌入基础模型，提高其在各种跨模态任务中的表现和适应性。

### 方法

提出SAIL-Embedding模型，采用多阶段训练方案：(1)内容感知渐进式训练增强模型对不同下游任务的适应性和跨模态能力；(2)协作感知推荐增强训练通过提取序列到项目和ID到项目嵌入知识并挖掘用户历史兴趣来优化推荐场景；(3)随机专业化和数据集驱动的模式匹配加强模型训练的灵活性和泛化能力。

### 主要发现

SAIL-Embedding在不同检索任务中实现了最先进性能；在抖音精选场景中，7天LT增长+0.158%，14天LT增长+0.144%；在抖音feed排序模型中，匹配特征带来+0.08%的AUC增益。

### 结论

SAIL-Embedding通过创新的训练策略和架构设计有效解决了现有多模态嵌入模型在实际应用中的局限性，显著提升了模型性能和业务指标。

### 翻译

多模态嵌入模型旨在产生信息丰富的统一表示，支持各种跨模态任务。尽管从基于CLIP的双塔架构到大型视觉语言模型的发展有前景，但先前的工作在实际应用和业务场景中仍面临不可避免的挑战，如模态支持有限、训练机制不稳定和工业领域差距。在这项工作中，我们引入了SAIL-Embedding，一个全模态嵌入基础模型，通过定制的训练策略和架构设计解决这些问题。在优化过程中，我们提出了多阶段训练方案，以提高表示学习的多方面有效性。具体而言，内容感知渐进式训练旨在增强模型对不同下游任务的适应性，掌握丰富的跨模态能力。协作感知推荐增强训练通过从序列到项目和ID到项目嵌入中提取知识，同时挖掘用户历史兴趣，进一步使多模态表示适应推荐场景。同时，我们开发了随机专业化和数据集驱动的模式匹配，以加强模型训练的灵活性和泛化能力。实验结果表明，与其他方法相比，SAIL-Embedding在不同的检索任务中实现了最先进的性能。在我们模型集成的各种现实场景的在线实验中，我们观察到Lifetime (LT)显著增加，这是推荐体验的关键指标。例如，在抖音精选场景中，模型实现了7天LT增长+0.158%和14天LT增长+0.144%。对于抖音feed排序模型，SAIL-Embedding产生的匹配特征带来了+0.08%的AUC增益。


### 论文摘要

Multimodal embedding models aim to yield informative unified representations that empower diverse cross-modal tasks. Despite promising developments in the evolution from CLIP-based dual-tower architectures to large vision-language models, prior works still face unavoidable challenges in real-world applications and business scenarios, such as the limited modality support, unstable training mechanisms, and industrial domain gaps. In this work, we introduce SAIL-Embedding, an omni-modal embedding foundation model that addresses these issues through tailored training strategies and architectural design. In the optimization procedure, we propose a multi-stage training scheme to boost the multifaceted effectiveness of representation learning. Specifically, the content-aware progressive training aims to enhance the model's adaptability to diverse downstream tasks and master enriched cross-modal proficiency. The collaboration-aware recommendation enhancement training further adapts multimodal representations for recommendation scenarios by distilling knowledge from sequence-to-item and ID-to-item embeddings while mining user historical interests. Concurrently, we develop the stochastic specialization and dataset-driven pattern matching to strengthen model training flexibility and generalizability. Experimental results show that SAIL-Embedding achieves SOTA performance compared to other methods in different retrieval tasks. In online experiments across various real-world scenarios integrated with our model, we observe a significant increase in Lifetime (LT), which is a crucial indicator for the recommendation experience. For instance, the model delivers the 7-day LT gain of +0.158% and the 14-day LT gain of +0.144% in the Douyin-Selected scenario. For the Douyin feed rank model, the match features produced by SAIL-Embedding yield a +0.08% AUC gain.

---

## 21. CoRA: Covariate-Aware Adaptation of Time Series Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.12681v1](http://arxiv.org/abs/2510.12681v1)

**作者:** Guo Qin, Zhi Chen, Yong Liu, Zhiyuan Shi, Haixuan Liu, Xiangdong Huang, Jianmin Wang, Mingsheng Long

**发布时间:** 2025-10-14

### GPT解析

### 总结

本文提出了一种协变量感知适应框架(CoRA)，用于增强时间序列基础模型(TSFMs)的性能，使其能够有效整合来自不同模态的外部协变量信息，显著提升预测质量。

### 背景

当前大多数TSFMs在单变量时间序列上进行预训练，这限制了它们在真实世界预测任务中利用不同协变量中的重要信息。这种限制源于变量间依赖性的异质性和在大规模多变量数据集上的骨干模型扩展性挑战。

### 目的

为了进一步提高TSFMs的性能，提出一个通用的协变量感知适应框架(CoRA)，使TSFMs能够有效整合来自时间序列、语言和图像等不同模态的外部协变量信息。

### 方法

CoRA框架利用预训练的基础模型骨干作为冻结特征提取器，采用Granger因果嵌入(GCE)自动评估协变量相对于目标变量的因果预测能力，并通过零初始化的条件注入机制整合加权嵌入，避免灾难性遗忘并逐渐融入外部信息。

### 主要发现

实验表明，TSFMs的CoRA超越了具有完全或少样本训练的最先进协变量感知深度预测器，在协变量感知预测上实现了31.1%的MSE降低。CoRA与各种先进TSFMs兼容性强，并将协变量范围扩展到其他模态。

### 结论

CoRA为TSFMs的应用提供了实用范式，能够有效整合多模态协变量信息，显著提升预测性能，扩展了TSFMs在实际应用中的能力范围。

### 翻译

时间序列基础模型(TSFMs)已通过其模型容量、可扩展性和零样本泛化能力显示出显著影响。然而，由于变量间依赖性的异质性和在大规模多变量数据集上的骨干模型扩展性，大多数TSFMs通常在单变量时间序列上进行预训练。这一限制使它们在真实世界预测任务中忽略了来自不同协变量的关键信息。为了进一步提高TSFMs的性能，我们提出了一个通用的协变量感知适应(CoRA)框架。它利用基础模型的预训练骨干，同时有效整合来自时间序列、语言和图像等不同模态的外部协变量，以提高预测质量。技术上，CoRA在适应过程中保持初始化等价性和参数一致性。将基础模型的骨干作为冻结特征提取器，实证表明基础模型的输出嵌入比原始数据更具信息量。此外，CoRA采用了一种新的Granger因果嵌入(GCE)来自动评估协变量相对于目标变量的因果预测能力。我们将这些加权嵌入与零初始化的条件注入机制相结合，避免了对预训练基础模型的灾难性遗忘，并逐渐整合外部信息。大量实验表明，TSFMs的CoRA超越了具有完全或少样本训练的最先进协变量感知深度预测器，在协变量感知预测上实现了31.1%的MSE降低。与其他适应方法相比，CoRA与各种先进的TSFMs具有强大的兼容性，并将协变量的范围扩展到其他模态，为TSFMs的应用提供了实用的范式。


### 论文摘要

Time Series Foundation Models (TSFMs) have shown significant impact through their model capacity, scalability, and zero-shot generalization. However, due to the heterogeneity of inter-variate dependencies and the backbone scalability on large-scale multivariate datasets, most TSFMs are typically pre-trained on univariate time series. This limitation renders them oblivious to crucial information from diverse covariates in real-world forecasting tasks. To further enhance the performance of TSFMs, we propose a general covariate-aware adaptation (CoRA) framework for TSFMs. It leverages pre-trained backbones of foundation models while effectively incorporating exogenous covariates from various modalities, including time series, language, and images, to improve the quality of predictions. Technically, CoRA maintains the equivalence of initialization and parameter consistency during adaptation. With preserved backbones of foundation models as frozen feature extractors, the outcome embeddings from foundation models are empirically demonstrated more informative than raw data. Further, CoRA employs a novel Granger Causality Embedding (GCE) to automatically evaluate covariates regarding their causal predictability with respect to the target variate. We incorporate these weighted embeddings with a zero-initialized condition-injection mechanism, avoiding catastrophic forgetting of pre-trained foundation models and gradually integrates exogenous information. Extensive experiments show that CoRA of TSFMs surpasses state-of-the-art covariate-aware deep forecasters with full or few-shot training samples, achieving 31.1% MSE reduction on covariate-aware forecasting. Compared to other adaptation methods, CoRA exhibits strong compatibility with various advanced TSFMs and extends the scope of covariates to other modalities, presenting a practical paradigm for the application of TSFMs.

---

## 22. On the Use of Hierarchical Vision Foundation Models for Low-Cost Human Mesh Recovery and Pose Estimation

**论文链接:** [http://arxiv.org/abs/2510.12660v1](http://arxiv.org/abs/2510.12660v1)

**作者:** Shuhei Tarashima, Yushan Wang, Norio Tagawa

**发布时间:** 2025-10-14

**备注:** Accepted at ICCVW 2025

### GPT解析

### 总结

本研究旨在开发简单高效的人体网格恢复(HMR)模型及其前置任务人体姿态估计(HPE)模型，通过利用分层视觉基础模型的早期阶段作为编码器，实现了在准确性和计算效率之间的更好权衡。

### 背景

当前最先进的HMR方法（如HMR2.0及其后续版本）依赖于大型、非分层的视觉Transformer作为编码器，这些编码器是从相应的HPE模型（如ViTPose）继承而来的。

### 目的

开发简单高效的人体网格恢复(HMR)模型及其前置任务人体姿态估计(HPE)模型。

### 方法

构建三种轻量级HMR2.0变体；提出利用Swin Transformer、GroupMixFormer和VMamba等分层视觉基础模型的早期阶段作为编码器；对27种基于分层VFMs的HMR和HPE模型进行全面评估。

### 主要发现

仅使用分层VFMs的前两或三个阶段就能达到与完整阶段模型相当的性能；截断后的模型在准确性和计算效率之间表现出比现有轻量级替代方案更好的权衡。

### 结论

通过利用分层视觉基础模型的早期阶段，可以开发出既高效又准确的人体网格恢复和姿态估计模型。

### 翻译

在这项工作中，我们旨在开发用于人体网格恢复(HMR)及其前置任务人体姿态估计(HPE)的简单高效模型。最先进的HMR方法（如HMR2.0及其后续版本）依赖于大型、非分层的视觉Transformer作为编码器，这些编码器是从相应的HPE模型（如ViTPose）继承而来的。为了在不同计算预算下建立基线，我们首先通过调整相应的ViTPose模型构建了三种轻量级HMR2.0变体。此外，我们提出利用Swin Transformer、GroupMixFormer和VMamba等分层视觉基础模型(VFMs)的早期阶段作为编码器。这种设计灵感来自于分层VFMs的中间阶段产生的特征图分辨率与非分层模型相当或更高。我们对27种基于分层VFMs的HMR和HPE模型进行了全面评估，证明仅使用前两或三个阶段就能达到与完整阶段模型相当的性能。此外，我们表明与现有的轻量级替代方案相比，截断后的模型在准确性和计算效率之间表现出更好的权衡。


### 论文摘要

In this work, we aim to develop simple and efficient models for human mesh recovery (HMR) and its predecessor task, human pose estimation (HPE). State-of-the-art HMR methods, such as HMR2.0 and its successors, rely on large, non-hierarchical vision transformers as encoders, which are inherited from the corresponding HPE models like ViTPose. To establish baselines across varying computational budgets, we first construct three lightweight HMR2.0 variants by adapting the corresponding ViTPose models. In addition, we propose leveraging the early stages of hierarchical vision foundation models (VFMs), including Swin Transformer, GroupMixFormer, and VMamba, as encoders. This design is motivated by the observation that intermediate stages of hierarchical VFMs produce feature maps with resolutions comparable to or higher than those of non-hierarchical counterparts. We conduct a comprehensive evaluation of 27 hierarchical-VFM-based HMR and HPE models, demonstrating that using only the first two or three stages achieves performance on par with full-stage models. Moreover, we show that the resulting truncated models exhibit better trade-offs between accuracy and computational efficiency compared to existing lightweight alternatives.

---

## 23. On Foundation Models for Temporal Point Processes to Accelerate Scientific Discovery

**论文链接:** [http://arxiv.org/abs/2510.12640v1](http://arxiv.org/abs/2510.12640v1)

**作者:** David Berghaus, Patrick Seifner, Kostadin Cvejoski, Ramses J. Sanchez

**发布时间:** 2025-10-14

### GPT解析

### 总结

论文介绍了一种基于基础模型的新方法，用于分析科学领域中的时间序列事件数据，无需为每个新数据集重新训练模型，从而加速科学发现。

### 背景

许多科学领域（从医学到地震学）依赖于分析随时间变化的事件序列来理解复杂系统。传统机器学习模型需要为每个新数据集从头构建和训练，这是一个缓慢且昂贵的过程。

### 目的

开发一种通用的事件分析方法，使复杂事件分析更加易于访问，并加快科学发现的步伐。

### 方法

创建一个单一的强大'基础模型'，在数百万个模拟事件序列上训练，学习事件数据的基本模式和事件如何展开的通用理解。

### 主要发现

该模型可以即时分析新的科学数据，无需重新训练，只需查看数据集中的几个示例；同时可以快速微调以获得更高的准确性。

### 结论

这种方法使复杂事件分析更加易于访问，并加速了科学发现的步伐。

### 翻译

许多科学领域，从医学到地震学，都依赖于分析随时间变化的事件序列来理解复杂系统。传统上，机器学习模型必须为每个新数据集从头构建和训练，这是一个缓慢且昂贵的过程。我们介绍了一种新方法：一个单一的、强大的模型，学习上下文中事件数据的基本模式。我们在数百万个模拟事件序列上训练了这个'基础模型'，教会它事件如何展开的通用理解。因此，我们的模型可以即时分析新的科学数据，无需重新训练，只需查看数据集中的几个示例。它还可以快速微调以获得更高的准确性。这种方法使复杂事件分析更加易于访问，并加快了科学发现的步伐。


### 论文摘要

Many scientific fields, from medicine to seismology, rely on analyzing sequences of events over time to understand complex systems. Traditionally, machine learning models must be built and trained from scratch for each new dataset, which is a slow and costly process. We introduce a new approach: a single, powerful model that learns the underlying patterns of event data in context. We trained this "foundation model" on millions of simulated event sequences, teaching it a general-purpose understanding of how events can unfold. As a result, our model can analyze new scientific data instantly, without retraining, simply by looking at a few examples from the dataset. It can also be quickly fine-tuned for even higher accuracy. This approach makes sophisticated event analysis more accessible and accelerates the pace of scientific discovery.

---

## 24. Unlocking Zero-Shot Plant Segmentation with Pl@ntNet Intelligence

**论文链接:** [http://arxiv.org/abs/2510.12579v1](http://arxiv.org/abs/2510.12579v1)

**作者:** Simon Ravé, Jean-Christophe Lombardo, Pejman Rasti, Alexis Joly, David Rousseau

**发布时间:** 2025-10-14

### GPT解析

### 总结

提出了一种结合Plantnet、DinoV2和SAM的农业图像零样本分割方法，利用Plantnet的植物识别能力生成粗略掩码，再通过SAM细化得到详细分割结果，无需收集新数据集。

### 背景

农业图像分割面临训练数据有限和田间条件复杂等挑战，这些因素常常阻碍纯监督方法的有效性，现有方法往往需要大量难以获取的标注数据。

### 目的

开发一种无需收集和标注新数据集的农业图像分割方法，解决农业场景中数据标注瓶颈问题，并在各种复杂农业场景中实现有效的分割。

### 方法

利用Plantnet（大型植物分类模型）及其DinoV2主干网络，结合Segment Anything Model (SAM)，使用Plantnet的专门植物表示来识别植物区域并生成粗略分割掩码，然后通过SAM进一步细化掩码以获得详细分割结果。

### 主要发现

使用Plantnet微调的DinoV2相比基础DinoV2模型在Jaccard指数(IoU)测量上展现出一致的性能提升，结合基础模型与专门的植物中心模型可以缓解标注瓶颈问题，并在各种农业场景中实现有效分割。

### 结论

将基础模型与专门的植物中心模型相结合具有潜力，可以减轻农业图像分割中的标注负担，并在多样化的农业场景中实现有效的分割结果。

### 翻译

我们提出了一种农业图像的零样本分割方法，该方法结合了Plantnet（一种大规模植物分类模型）、其DinoV2主干网络和Segment Anything Model (SAM)。我们无需收集和标注新数据集，而是利用Plantnet专门的植物表示来识别植物区域并生成粗略分割掩码。然后，这些掩码通过SAM进行细化以产生详细分割结果。我们在四个公开可用数据集上进行了评估，这些数据集在对比度方面具有不同复杂度，包括一些训练数据有限且田间条件复杂常常阻碍纯监督方法的数据集。我们的结果显示，与基础DinoV2模型相比，使用Plantnet微调的DinoV2在Jaccard指数(IoU)测量上展现出一致的性能提升。这些发现强调了将基础模型与专门的植物中心模型相结合的潜力，可以减轻标注瓶颈，并在多样化的农业场景中实现有效分割。


### 论文摘要

We present a zero-shot segmentation approach for agricultural imagery that leverages Plantnet, a large-scale plant classification model, in conjunction with its DinoV2 backbone and the Segment Anything Model (SAM). Rather than collecting and annotating new datasets, our method exploits Plantnet's specialized plant representations to identify plant regions and produce coarse segmentation masks. These masks are then refined by SAM to yield detailed segmentations. We evaluate on four publicly available datasets of various complexity in terms of contrast including some where the limited size of the training data and complex field conditions often hinder purely supervised methods. Our results show consistent performance gains when using Plantnet-fine-tuned DinoV2 over the base DinoV2 model, as measured by the Jaccard Index (IoU). These findings highlight the potential of combining foundation models with specialized plant-centric models to alleviate the annotation bottleneck and enable effective segmentation in diverse agricultural scenarios.

---

## 25. HEAR: An EEG Foundation Model with Heterogeneous Electrode Adaptive Representation

**论文链接:** [http://arxiv.org/abs/2510.12515v1](http://arxiv.org/abs/2510.12515v1)

**作者:** Zhige Chen, Chengxuan Qin, Wenlong You, Rui Liu, Congying Chu, Rui Yang, Kay Chen Tan, Jibin Wu

**发布时间:** 2025-10-14

### GPT解析

### 总结

本文介绍了HEAR，这是首个专门设计用于支持异构EEG设备的EEG基础模型，能够适应不同的电极布局和电极数量，通过可学习的基于坐标的空间嵌入和空间引导transformer实现统一表示空间，实验结果表明HEAR在支持异构EEG设备和跨任务跨主体泛化方面显著优于现有模型。

### 背景

EEG是神经科学研究和脑机接口应用的重要技术，近期开发的大规模EEG基础模型展现出强大的跨任务和跨主体泛化能力，但EEG设备的异质性阻碍了这些模型的广泛采用和进一步发展。

### 目的

开发一种能够支持异构EEG设备的EEG基础模型，解决不同电极布局和电极数量带来的兼容性问题。

### 方法

HEAR采用可学习的基于坐标的空间嵌入将不同布局和数量的电极映射到统一表示空间，并通过空间引导transformer处理这些表示以捕获电极间的时空依赖关系；同时构建了一个包含8,782小时数据、来自150多种电极布局的数据集来支持模型开发。

### 主要发现

实验结果表明HEAR在支持异构EEG设备方面显著优于现有EEG基础模型，并在多样化的认知任务和主体间表现出良好的泛化能力。

### 结论

HEAR为解决EEG设备异质性挑战提供了有效方法，有助于EEG基础模型的广泛应用和进一步发展。

### 翻译

脑电图是神经科学研究和脑机接口应用的关键技术。最近，大规模EEG基础模型已被开发，展现出跨不同任务和主体的强大泛化能力。然而，EEG设备的异质性不仅阻碍了这些模型的广泛采用，也对其进一步扩展和发展提出了重大挑战。在本文中，我们介绍了HEAR，这是首个专门设计用于支持异构EEG设备的EEG基础模型，能够适应不同的电极布局和电极数量。HEAR采用可学习的基于坐标的空间嵌入，将具有不同布局和数量的电极映射到统一的表示空间。然后，这种统一的空间表示由新颖的空间引导transformer处理，有效捕获了电极间的时空依赖关系。为支持HEAR的开发，我们构建了一个包含8,782小时数据的大规模EEG数据集，数据来自150多种不同的电极布局，电极数量最多达1,132个。实验结果表明，HEAR在支持异构EEG设备方面显著优于现有的EEG基础模型，并能很好地泛化到多样化的认知任务和主体中。


### 论文摘要

Electroencephalography (EEG) is an essential technique for neuroscience research and brain-computer interface (BCI) applications. Recently, large-scale EEG foundation models have been developed, exhibiting robust generalization capabilities across diverse tasks and subjects. However, the heterogeneity of EEG devices not only hinders the widespread adoption of these models but also poses significant challenges to their further scaling and development. In this paper, we introduce HEAR, the first EEG foundation model explicitly designed to support heterogeneous EEG devices, accommodating varying electrode layouts and electrode counts. HEAR employs a learnable, coordinate-based spatial embedding to map electrodes with diverse layouts and varying counts into a unified representational space. This unified spatial representation is then processed by a novel spatially-guided transformer, which effectively captures spatiotemporal dependencies across electrodes. To support the development of HEAR, we construct a large-scale EEG dataset comprising 8,782 hours of data collected from over 150 distinct electrode layouts with up to 1,132 electrodes. Experimental results demonstrate that HEAR substantially outperforms existing EEG foundation models in supporting heterogeneous EEG devices and generalizing across diverse cognitive tasks and subjects.

---

## 26. Artificial Intelligence Virtual Cells: From Measurements to Decisions across Modality, Scale, Dynamics, and Evaluation

**论文链接:** [http://arxiv.org/abs/2510.12498v1](http://arxiv.org/abs/2510.12498v1)

**作者:** Chengpeng Hu, Calvin Yu-Chian Chen

**发布时间:** 2025-10-14

### GPT解析

### 总结

人工智能虚拟细胞(AIVCs)旨在从多模态、多尺度数据中学习细胞状态模型，当前研究面临跨实验室迁移性有限、数据分割偏差、剂量时间效应处理不足以及跨尺度耦合受限等挑战。作者提出细胞状态潜在(CSL)视角，通过操作符语法组织学习，并建议改进评估方法和数据设计。

### 背景

AIVCs致力于从多模态、多尺度测量中学习可执行的、决策相关的细胞状态模型。近期研究已引入单细胞和空间基础模型，改进跨模态对齐，扩展扰动图谱，并探索通路水平读出。

### 目的

提出一种与模型无关的细胞状态潜在(CSL)视角，通过操作符语法组织学习，并建立跨模态、尺度、背景和干预的决策对齐评估蓝图。

### 方法

采用操作符语法组织学习：测量、提升/投影(用于跨尺度耦合)和干预(用于剂量和调度)。强调功能空间读出，如通路活性、空间邻域和临床相关终点。

### 主要发现

当前评估主要局限于单个数据集和设置；跨实验室和平台的可迁移性有限；某些数据分割易受泄漏和覆盖偏差影响；剂量、时间和组合效应未得到系统处理；跨尺度耦合受限，分子、细胞和组织水平的锚点稀少。

### 结论

建议采用操作符感知的数据设计、抗泄漏分区和透明校准与报告，以实现可重复的、一对一的比较，改进AIVCs的评估方法。

### 翻译

人工智能虚拟细胞(AIVCs)旨在从多模态、多尺度测量中学习可执行的、决策相关的细胞状态模型。近期研究已引入单细胞和空间基础模型，改进跨模态对齐，扩展扰动图谱，并探索通路水平读出。然而，尽管保留验证是标准实践，评估仍主要局限于单个数据集和设置；证据表明跨实验室和平台的可迁移性通常有限，某些数据分割易受泄漏和覆盖偏差影响，剂量、时间和组合效应尚未得到系统处理。跨尺度耦合仍然受限，因为连接分子、细胞和组织水平的锚点稀少，且与科学或临床读出的对齐在不同研究中各不相同。我们提出了一种与模型无关的细胞状态潜在(CSL)视角，通过操作符语法组织学习：测量、提升/投影(用于跨尺度耦合)和干预(用于剂量和调度)。这一观点激发了跨模态、尺度、背景和干预的决策对齐评估蓝图，并强调功能空间读出，如通路活性、空间邻域和临床相关终点。我们建议采用操作符感知的数据设计、抗泄漏分区和透明校准与报告，以实现可重复的、一对一的比较。


### 论文摘要

Artificial Intelligence Virtual Cells (AIVCs) aim to learn executable, decision-relevant models of cell state from multimodal, multiscale measurements. Recent studies have introduced single-cell and spatial foundation models, improved cross-modality alignment, scaled perturbation atlases, and explored pathway-level readouts. Nevertheless, although held-out validation is standard practice, evaluations remain predominantly within single datasets and settings; evidence indicates that transport across laboratories and platforms is often limited, that some data splits are vulnerable to leakage and coverage bias, and that dose, time and combination effects are not yet systematically handled. Cross-scale coupling also remains constrained, as anchors linking molecular, cellular and tissue levels are sparse, and alignment to scientific or clinical readouts varies across studies. We propose a model-agnostic Cell-State Latent (CSL) perspective that organizes learning via an operator grammar: measurement, lift/project for cross-scale coupling, and intervention for dosing and scheduling. This view motivates a decision-aligned evaluation blueprint across modality, scale, context and intervention, and emphasizes function-space readouts such as pathway activity, spatial neighborhoods and clinically relevant endpoints. We recommend operator-aware data design, leakage-resistant partitions, and transparent calibration and reporting to enable reproducible, like-for-like comparisons.

---

## 27. A Hierarchical Quantized Tokenization Framework for Task-Adaptive Graph Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.12369v1](http://arxiv.org/abs/2510.12369v1)

**作者:** Yang Xiang, Li Fan, Chenke Yin, Chengtao Ji

**发布时间:** 2025-10-14

### GPT解析

### 总结

本文提出了一种分层量化框架，通过自加权机制实现跨尺度的任务自适应聚合，在保持编码器冻结的同时，通过轻量级门控过程调节信息流，实现参数高效的下游任务适应。

### 背景

语言和视觉基础模型的进展表明，将复杂输入转换为紧凑序列的离散标记接口对大规模建模至关重要。将此范式扩展到图需要处理非欧几里得结构和多尺度依赖关系的标记化方案。

### 目的

解决现有图标记化方法（线性化、连续和量化）在适应性和效率上的局限性，特别是解决基于量化的标记化方法在组织分层信息时缺乏任务自适应性的问题。

### 方法

提出了一种分层量化框架，引入自加权机制用于跨尺度的任务自适应聚合。该方法保持编码器冻结，通过轻量级门控过程调节信息流，实现参数高效的下游任务适应。

### 主要发现

在节点分类和链接预测的基准数据集上，所提出的方法在可比的计算预算下，与强大的基线方法相比取得了持续改进。

### 结论

该分层量化框架能够有效处理非欧几里得结构和多尺度依赖关系，通过自加权机制实现任务自适应信息聚合，为图建模提供了新的高效解决方案。

### 翻译

语言和视觉基础模型的最新进展表明，将复杂输入转换为紧凑序列的离散标记接口对大规模建模至关重要。将这一范式扩展到图需要一种能够高效处理非欧几里得结构和多尺度依赖关系的标记化方案。现有的图标记化方法，包括线性化、连续和量化方法，在适应性和效率方面仍然存在局限性。特别是，大多数当前基于量化的标记化方法以固定或任务无关的方式组织分层信息，这可能导致过度表示或未充分利用结构线索，并且无法在不重新编码器的情况下动态重新加权不同级别的贡献。本文提出了一种分层量化框架，引入了跨多个尺度进行任务自适应聚合的自加权机制。所提出的方法保持编码器冻结，同时通过轻量级门控过程调节信息流，实现参数高效地适应多样化的下游任务。在节点分类和链接预测的基准数据集上的实验表明，在可比的计算预算下，该方法比强大的基线方法取得了持续改进。


### 论文摘要

Recent progress in language and vision foundation models demonstrates the importance of discrete token interfaces that transform complex inputs into compact sequences for large-scale modeling. Extending this paradigm to graphs requires a tokenization scheme that handles non-Euclidean structures and multi-scale dependencies efficiently. Existing approaches to graph tokenization, linearized, continuous, and quantized, remain limited in adaptability and efficiency. In particular, most current quantization-based tokenizers organize hierarchical information in fixed or task-agnostic ways, which may either over-represent or under-utilize structural cues, and lack the ability to dynamically reweight contributions from different levels without retraining the encoder. This work presents a hierarchical quantization framework that introduces a self-weighted mechanism for task-adaptive aggregation across multiple scales. The proposed method maintains a frozen encoder while modulating information flow through a lightweight gating process, enabling parameter-efficient adaptation to diverse downstream tasks. Experiments on benchmark datasets for node classification and link prediction demonstrate consistent improvements over strong baselines under comparable computational budgets.

---

## 28. DeePAQ: A Perceptual Audio Quality Metric Based On Foundational Models and Weakly Supervised Learning

**论文链接:** [http://arxiv.org/abs/2510.12326v1](http://arxiv.org/abs/2510.12326v1)

**作者:** Guanxin Jiang, Andreas Brendel, Pablo M. Delgado, Jürgen Herre

**发布时间:** 2025-10-14

**备注:** 5 pages, 2 figures

### GPT解析

### 总结

该研究提出了DeePAQ，一种基于深度学习的感知音频质量评估指标，用于评估通用音频质量。该方法结合了度量学习和音乐基础模型MERT，通过代理标签指导构建捕获音频失真强度的嵌入空间。研究首次在通用音频质量领域利用弱监督标签和度量学习微调音乐基础模型，使用低秩适应(LoRA)技术。实验表明，该方法在检测编码伪影方面优于现有指标，并能良好泛化到未见过的失真类型。

### 背景

在音频质量评估领域，需要能够准确评估通用音频质量的指标。现有方法可能在处理不同类型的音频失真时存在局限性，特别是在编码伪影和源分离等场景中。

### 目的

开发一种新的音频质量评估指标，能够准确捕捉通用音频中的失真强度，并在多种音频处理场景中表现良好。

### 方法

研究采用度量学习结合音乐基础模型MERT的方法，通过代理标签指导，构建一个能够捕获通用音频失真强度的嵌入空间。该方法创新性地在通用音频质量领域应用弱监督标签和度量学习来微调音乐基础模型，并使用低秩适应(LoRA)技术进行参数高效调整。

### 主要发现

在音频编码和源分离的听力测试中，DeePAQ超越了现有的最先进目标音频质量指标。特别是在检测编码伪影方面表现优异，并且对未见过的失真类型（如源分离）具有良好的泛化能力，展示了其鲁棒性和多功能性。

### 结论

DeePAQ是一种创新的音频质量评估方法，通过结合度量学习和音乐基础模型，有效地解决了通用音频质量评估的挑战。其优越的性能和泛化能力表明该方法在音频处理领域具有广泛的应用前景。

### 翻译

本文提出了用于评估通用音频质量的基于深度学习的感知音频质量指标(DeePAQ)。我们的方法结合了度量学习和音乐基础模型MERT，通过代理标签指导，构建一个捕获通用音频中失真强度的嵌入空间。据我们所知，DeePAQ是通用音频质量领域中首个利用弱监督标签和度量学习来使用低秩适应(LoRA)微调音乐基础模型的方法，这一方向尚未被其他最先进方法探索。我们在涵盖音频编码和源分离的听力测试中，将所提出模型与最先进的目标音频质量指标进行了基准测试。结果表明，我们的方法在检测编码伪影方面超越了现有指标，并且对源分离等未见过的失真具有良好的泛化能力，突显了其鲁棒性和多功能性。


### 论文摘要

This paper presents the Deep learning-based Perceptual Audio Quality metric (DeePAQ) for evaluating general audio quality. Our approach leverages metric learning together with the music foundation model MERT, guided by surrogate labels, to construct an embedding space that captures distortion intensity in general audio. To the best of our knowledge, DeePAQ is the first in the general audio quality domain to leverage weakly supervised labels and metric learning for fine-tuning a music foundation model with Low-Rank Adaptation (LoRA), a direction not yet explored by other state-of-the-art methods. We benchmark the proposed model against state-of-the-art objective audio quality metrics across listening tests spanning audio coding and source separation. Results show that our method surpasses existing metrics in detecting coding artifacts and generalizes well to unseen distortions such as source separation, highlighting its robustness and versatility.

---

## 29. Spatial Forcing: Implicit Spatial Representation Alignment for Vision-language-action Model

**论文链接:** [http://arxiv.org/abs/2510.12276v1](http://arxiv.org/abs/2510.12276v1)

**作者:** Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, Haoang Li

**发布时间:** 2025-10-14

### GPT解析

### 总结

本研究提出了空间强制(Spatial Forcing, SF)策略，一种简单有效的对齐方法，使视觉-语言-动作(VLA)模型能够隐式发展空间理解能力，无需依赖显式3D输入或深度估计器。

### 背景

视觉-语言-动作(VLA)模型在机器人执行语言指令和精确动作方面显示出强大潜力，但大多数VLA模型基于仅在2D数据上预训练的视觉-语言模型构建，缺乏准确的空间感知能力，限制了它们在3D物理世界中的操作能力。

### 目的

提出一种简单有效的对齐策略，使VLA模型能够在不依赖显式3D输入或深度估计器的情况下隐式发展空间理解能力，从而提高动作精确度。

### 方法

提出了空间强制(SF)策略，将VLA的中间视觉嵌入与预训练的3D基础模型产生的几何表示对齐。通过在中间层强制对齐，引导VLA编码更丰富的空间表示。

### 主要发现

在模拟和真实环境中的大量实验表明，SF实现了最先进的结果，超越了基于2D和3D的VLA模型。SF最多可加速训练3.8倍，并在多样化的机器人任务中提高了数据效率。

### 结论

SF是一种有效的策略，可以增强VLA模型的空间理解能力，不需要显式的3D输入或深度估计器，在性能、训练速度和数据效率方面都有显著提升。

### 翻译

视觉-语言-动作(VLA)模型最近在使机器人能够遵循语言指令并执行精确动作方面显示出强大潜力。然而，大多数VLA模型构建于仅在2D数据上预训练的视觉-语言模型之上，这些模型缺乏准确的空间感知能力，阻碍了它们在3D物理世界中的操作能力。现有解决方案尝试整合显式的3D传感器输入，如深度图或点云，但由于传感器噪声、硬件异构性和现有数据集中深度覆盖不完整，这些方法面临挑战。从2D图像估计3D线索的替代方法也受到深度估计器性能有限的困扰。我们提出了空间强制(SF)，一种简单而有效的对齐策略，隐式地强制VLA模型发展空间理解能力，而不依赖显式3D输入或深度估计器。SF将VLA的中间视觉嵌入与预训练的3D基础模型产生的几何表示对齐。通过在中间层强制对齐，SF引导VLA编码更丰富的空间表示，从而提高动作精确度。在模拟和真实环境中的大量实验表明，SF实现了最先进的结果，超越了基于2D和3D的VLA。SF最多可加速训练3.8倍，并在多样化的机器人任务中提高了数据效率。项目页面位于https://spatial-forcing.github.io/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉-语言-行动模型缺乏准确空间感知能力的问题。大多数VLA模型构建于仅在2D数据上预训练的视觉-语言模型之上，难以在3D物理世界中有效操作。这个问题很重要，因为机器人操作需要将语义推理与3D物理世界的精确空间控制相结合，缺乏空间感知能力会阻碍机器人在真实世界中的任务执行。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过深度探测实验发现现有VLA模型的视觉嵌入无法产生有意义的空间结构，从而假设需要隐式地增强模型的空间感知能力。作者借鉴了表示监督（representation supervision）的思路，特别是受到ROSS、REPA等工作的启发，采用表示对齐的方法。同时，作者利用了VGGT作为预训练的3D基础模型来生成空间表示，并借鉴了VLA模型中自回归机制的设计。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过空间强制（SF）策略隐式地强制VLA模型发展空间理解能力，而不依赖显式的3D输入。实现流程包括：1) 将多视角图像输入到VGGT 3D基础模型生成空间表示；2) 添加位置嵌入保留空间顺序；3) 使用余弦相似度对齐VLA的视觉标记与空间表示；4) 监督较深但不是最深的层（如第24层）；5) 结合动作生成损失和对齐损失进行训练；6) 推理时与标准VLA相同，无额外计算开销。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出空间强制（SF）对齐策略；2) 不依赖显式3D输入或深度估计器；3) 利用VGGT保证多视角一致性；4) 发现特定层（第24层）监督最有效。相比之前工作，SF避免了3D传感器噪声和硬件异构性问题，不依赖深度估计器性能限制，不仅提高了性能，还加速了训练（最高3.8倍）并提高了数据效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了空间强制策略，通过隐式对齐VLA模型的视觉嵌入与3D基础模型的空间表示，在不依赖显式3D输入的情况下，显著提升了模型的空间感知能力、训练效率和数据效率。'}


### 论文摘要

Vision-language-action (VLA) models have recently shown strong potential in enabling robots to follow language instructions and execute precise actions. However, most VLAs are built upon vision-language models pretrained solely on 2D data, which lack accurate spatial awareness and hinder their ability to operate in the 3D physical world. Existing solutions attempt to incorporate explicit 3D sensor inputs such as depth maps or point clouds, but these approaches face challenges due to sensor noise, hardware heterogeneity, and incomplete depth coverage in existing datasets. Alternative methods that estimate 3D cues from 2D images also suffer from the limited performance of depth estimators.We propose Spatial Forcing (SF), a simple yet effective alignment strategy that implicitly forces VLA models to develop spatial comprehension capabilities without relying on explicit 3D inputs or depth estimators. SF aligns intermediate visual embeddings of VLAs with geometric representations produced by pretrained 3D foundation models. By enforcing alignment at intermediate layers, SF guides VLAs to encode richer spatial representations that enhance action precision.Extensive experiments in simulation and real-world environments demonstrate that SF achieves state-of-the-art results, surpassing both 2D- and 3D-based VLAs. SF further accelerates training by up to 3.8x and improves data efficiency across diverse robotic tasks. Project page is at https://spatial-forcing.github.io/

---

## 30. Evolution of meta's llama models and parameter-efficient fine-tuning of large language models: a survey

**论文链接:** [http://arxiv.org/abs/2510.12178v1](http://arxiv.org/abs/2510.12178v1)

**作者:** Abdulhady Abas Abdullah, Arkaitz Zubiaga, Seyedali Mirjalili, Amir H. Gandomi, Fatemeh Daneshfar, Mohammadsadra Amini, Alan Salam Mohammed, Hadi Veisi

**发布时间:** 2025-10-14

### GPT解析

### 总结

这篇综述论文全面介绍了Meta AI的LLaMA系列模型（从LLaMA 1到LLaMA 4）及其参数高效微调(PEFT)方法的发展历程，为机器学习研究人员和实践者提供了一站式资源。

### 背景

Meta AI的LLaMA系列模型经历了快速演进，从最初的LLaMA 1发展到LLaMA 4，同时针对这些模型开发了一系列专门的参数高效微调方法。

### 目的

提供对LLaMA模型家族和PEFT方法的全面概述，包括模型架构、性能特征、微调方法及其应用，帮助研究人员和实践者了解这一领域的最新进展。

### 方法

描述了LLaMA基础模型（参数量从7B-65B到288B）、架构（包括原生多模态和专家混合变体）和关键性能特征；讨论了PEFT概念和五种应用于LLaMA的PEFT方法；分析了模型和适配器架构、参数量和基准测试结果；考察了实际应用案例。

### 主要发现

详细讨论了LoRA、LLaMA-Adapter V1和V2、LLaMA-Excitor和QLoRA等PEFT方法的机制、参数节省和应用示例；展示了微调后的LLaMA模型在某些任务上优于更大的基线模型；总结了LLaMA模型和PEFT在法律和医疗等领域的成功应用。

### 结论

指出了当前面临的挑战和未来研究方向，如扩展到更大的上下文窗口和改进模型鲁棒性等，为后续研究提供了指导方向。

### 翻译

本综述回顾了Meta AI的LLaMA（大型语言模型Meta AI）系列的快速演进历程 - 从LLaMA 1到LLaMA 4，以及为这些模型开发的专门参数高效微调(PEFT)方法。我们首先描述了LLaMA基础模型家族（7B-65B到288B参数）、它们的架构（包括原生多模态和专家混合变体）以及关键性能特征。然后我们描述并讨论了PEFT概念，它通过仅更新一小部分参数来适应大型预训练模型，并回顾了五种已应用于LLaMA的PEFT方法：LoRA（低秩自适应）、LLaMA-Adapter V1和V2、LLaMA-Excitor和QLoRA（量化LoRA）。我们讨论了每种方法的机制、参数节省以及在LLaMA上的应用示例（如指令微调、多模态任务）。我们对模型和适配器架构、参数量和基准测试结果进行了结构化讨论和分析（包括微调后的LLaMA模型优于更大基线模型的示例）。最后，我们考察了LLaMA模型和PEFT已成功应用的实际情况（如法律和医疗领域），并讨论了当前面临的挑战和未来研究方向（如扩展到更大的上下文和改进鲁棒性）。这篇综述论文为对LLaMA模型和高效微调策略感兴趣的机器学习研究人员和实践者提供了一站式资源。


### 论文摘要

This review surveys the rapid evolution of Meta AI's LLaMA (Large Language Model Meta AI) series - from LLaMA 1 through LLaMA 4 and the specialized parameter-efficient fine-tuning (PEFT) methods developed for these models. We first describe the LLaMA family of foundation models (7B-65B to 288B parameters), their architectures (including native multimodal and Mixtureof-Experts variants), and key performance characteristics. We then describe and discuss the concept of PEFT, which adapts large pre-trained models by updating only a small subset of parameters, and review five PEFT methods that have been applied to LLaMA: LoRA (Low-Rank Adaptation), LLaMA-Adapter V1 and V2, LLaMA-Excitor, and QLoRA (Quantized LoRA). We discuss each method's mechanism, parameter savings, and example application to LLaMA (e.g., instruction tuning, multimodal tasks). We provide structured discussion and analysis of model and adapter architectures, parameter counts, and benchmark results (including examples where fine-tuned LLaMA models outperform larger baselines). Finally, we examine real-world use cases where LLaMA-based models and PEFT have been successfully applied (e.g., legal and medical domains), and we discuss ongoing challenges and future research directions (such as scaling to even larger contexts and improving robustness). This survey paper provides a one-stop resource for ML researchers and practitioners interested in LLaMA models and efficient fine-tuning strategies.

---

## 31. Playmate2: Training-Free Multi-Character Audio-Driven Animation via Diffusion Transformer with Reward Feedback

**论文链接:** [http://arxiv.org/abs/2510.12089v1](http://arxiv.org/abs/2510.12089v1)

**作者:** Xingpei Ma, Shenneng Huang, Jiaran Cai, Yuansheng Guan, Shen Zheng, Hanfeng Zhao, Qiang Zhang, Shunsi Zhang

**发布时间:** 2025-10-14

### GPT解析

### 总结

论文提出了一个基于DiT的框架和一种无需训练的多角色音频驱动动画方法，解决了现有技术在口型同步、长视频连贯性和多角色动画方面的挑战，实现了高质量、时间连贯且支持多角色的音频驱动视频生成。

### 背景

扩散模型的最新进展显著提高了音频驱动人体视频生成的质量和可控性，超越了传统方法。然而，现有方法仍面临口型同步准确性、长视频生成的时间连贯性以及多角色动画的挑战。

### 目的

开发一个能够生成任意长度逼真说话视频的框架，以及一种无需训练的多角色音频驱动动画方法，以解决现有技术面临的挑战。

### 方法

采用基于LoRA的训练策略结合位置推理方法实现高效长视频生成；结合部分参数更新与奖励反馈增强口型同步和自然身体运动；提出无需训练的Mask分类器自由引导方法用于多角色动画，支持三个或更多角色的音频驱动动画。

### 主要发现

实验结果表明，该方法优于现有的最先进方法，以简单、高效和经济的方式实现了高质量、时间连贯且支持多角色的音频驱动视频生成。

### 结论

所提出的DiT框架和Mask-CFG方法有效解决了音频驱动人体视频生成中的关键挑战，为高质量、时间连贯的多角色视频生成提供了新途径。

### 翻译

扩散模型的最新进展显著提高了音频驱动人体视频生成的质量，在质量和可控性方面超越了传统方法。然而，现有方法仍面临口型同步准确性、长视频生成的时间连贯性以及多角色动画的挑战。在这项工作中，我们提出了一个基于扩散变换器(DiT)的框架，用于生成任意长度的逼真说话视频，并引入了一种无需训练的多角色音频驱动动画方法。首先，我们采用基于LoRA的训练策略结合位置推理方法，使能够高效生成长视频同时保留基础模型能力。此外，我们将部分参数更新与奖励反馈相结合，以增强口型同步和自然的身体运动。最后，我们提出了一种无需训练的方法，即掩码分类器自由引导(Mask-CFG)，用于多角色动画，这不需要专门的数据集或模型修改，并支持三个或更多角色的音频驱动动画。实验结果表明，我们的方法优于现有的最先进方法，以简单、高效和经济的方式实现了高质量、时间连贯且支持多角色的音频驱动视频生成。


### 论文摘要

Recent advances in diffusion models have significantly improved audio-driven human video generation, surpassing traditional methods in both quality and controllability. However, existing approaches still face challenges in lip-sync accuracy, temporal coherence for long video generation, and multi-character animation. In this work, we propose a diffusion transformer (DiT)-based framework for generating lifelike talking videos of arbitrary length, and introduce a training-free method for multi-character audio-driven animation. First, we employ a LoRA-based training strategy combined with a position shift inference approach, which enables efficient long video generation while preserving the capabilities of the foundation model. Moreover, we combine partial parameter updates with reward feedback to enhance both lip synchronization and natural body motion. Finally, we propose a training-free approach, Mask Classifier-Free Guidance (Mask-CFG), for multi-character animation, which requires no specialized datasets or model modifications and supports audio-driven animation for three or more characters. Experimental results demonstrate that our method outperforms existing state-of-the-art approaches, achieving high-quality, temporally coherent, and multi-character audio-driven video generation in a simple, efficient, and cost-effective manner.

---

## 32. Conjecturing: An Overlooked Step in Formal Mathematical Reasoning

**论文链接:** [http://arxiv.org/abs/2510.11986v1](http://arxiv.org/abs/2510.11986v1)

**作者:** Jasivan Alex Sivakumar, Philipp Borchert, Ronald Cardenas, Gerasimos Lampouras

**发布时间:** 2025-10-13

### GPT解析

### 总结

本文研究了数学自动形式化中的猜想步骤，创建了ConjectureBench数据集和评估框架，发现LLMs的自动形式化性能被高估，并提出Lean-FIRe方法实现了PutnamBench问题的端到端自动形式化。

### 背景

自动形式化通常被视为直接翻译过程，忽略了关键的猜想步骤。许多数学问题需要先做出猜想才能形式化，而LLMs在自动形式化方面已存在困难，且对它们猜想能力的评估有限且常与自动形式化或证明纠缠。

### 目的

创建专门评估LLMs猜想能力的框架，既作为独立任务也作为自动形式化流程的一部分，探究猜想对自动形式化的影响，并提出改进方法。

### 方法

创建ConjectureBench数据集，重新设计评估框架和指标评估LLMs的猜想能力，设计Lean-FIRe推理时间方法改进猜想和自动形式化性能。

### 主要发现

当评估中考虑猜想时，GPT-4.1和DeepSeek-V3.1等基础模型的自动形式化性能被大大高估；Lean-FIRe方法首次实现了PutnamBench中13个问题(GPT-4.1)和7个问题(DeepSeek-V3.1)的端到端自动形式化。

### 结论

虽然LLMs拥有生成准确猜想所需的知识，但提高自动形式化性能需要将猜想视为独立任务，并研究如何将其正确整合到自动形式化中。

### 翻译

自动形式化是将非正式数学语言表达为正式数学语言的任务，通常被视为直接翻译过程。然而，这忽略了一个关键的先行步骤：猜想。许多数学问题不能直接形式化，需要先做出结论性猜想，如明确答案或特定界限。由于大型语言模型已经难以进行自动形式化，且对其猜想能力的评估有限且常与自动形式化或证明纠缠在一起，理解其影响尤其具有挑战性。为解决这一差距，我们扩充现有数据集创建了ConjectureBench，并重新设计了评估框架和指标，专门用于测量LLMs的猜想能力，既作为独立任务，也作为自动形式化流程的一部分。我们对基础模型的评估（包括GPT-4.1和DeepSeek-V3.1）显示，当评估中考虑猜想时，它们的自动形式化性能被大大高估。然而，不应假设猜想会预先提供。我们设计了一种推理时间方法Lean-FIRe来改进猜想和自动形式化，据我们所知，这首次实现了GPT-4.1对13个PutnamBench问题和DeepSeek-V3.1对7个问题的端到端自动形式化。我们证明，虽然LLMs拥有生成准确猜想所需的知识，但提高自动形式化性能需要将猜想视为独立任务，并进一步研究如何将其正确整合到自动形式化中。最后，我们提供前瞻性指导，引导未来研究关注改进猜想这一被忽视的正式数学推理步骤。


### 论文摘要

Autoformalisation, the task of expressing informal mathematical statements in formal language, is often viewed as a direct translation process. This, however, disregards a critical preceding step: conjecturing. Many mathematical problems cannot be formalised directly without first conjecturing a conclusion such as an explicit answer, or a specific bound. Since Large Language Models (LLMs) already struggle with autoformalisation, and the evaluation of their conjecturing ability is limited and often entangled within autoformalisation or proof, it is particularly challenging to understand its effect. To address this gap, we augment existing datasets to create ConjectureBench, and redesign the evaluation framework and metric specifically to measure the conjecturing capabilities of LLMs both as a distinct task and within the autoformalisation pipeline. Our evaluation of foundational models, including GPT-4.1 and DeepSeek-V3.1, reveals that their autoformalisation performance is substantially overestimated when the conjecture is accounted for during evaluation. However, the conjecture should not be assumed to be provided. We design an inference-time method, Lean-FIRe to improve conjecturing and autoformalisation, which, to the best of our knowledge, achieves the first successful end-to-end autoformalisation of 13 PutnamBench problems with GPT-4.1 and 7 with DeepSeek-V3.1. We demonstrate that while LLMs possess the requisite knowledge to generate accurate conjectures, improving autoformalisation performance requires treating conjecturing as an independent task, and investigating further how to correctly integrate it within autoformalisation. Finally, we provide forward-looking guidance to steer future research toward improving conjecturing, an overlooked step of formal mathematical reasoning.

---

## 33. Balancing Synthetic Data and Replay for Enhancing Task-Specific Capabilities

**论文链接:** [http://arxiv.org/abs/2510.11842v1](http://arxiv.org/abs/2510.11842v1)

**作者:** Urs Spiegelhalter, Jörg K. H. Franke, Frank Hutter

**发布时间:** 2025-10-13

**备注:** Presented at 39th Conference on Neural Information Processing Systems  (NeurIPS 2025) Workshop on Continual and Compatible Foundation Model Updates  (CCFM)

### GPT解析

### 总结

这项研究探讨了在语言模型持续预训练适应新任务时，如何平衡任务性能和知识保留的问题，特别关注了重放比率配置与计算预算之间的相互作用，并提供了基于实证的选择指南。

### 背景

通过持续预训练使语言模型适应新任务面临一个基本权衡：模型必须学习新能力，同时避免对现有知识的灾难性遗忘。先前的研究已经研究了合成数据生成技术，但在计算约束下平衡任务性能和知识保留的最佳重放比率仍不清楚。

### 目的

研究重放比率配置与计算预算在语言模型适应新任务时的相互作用，分析不同总令牌预算和重放比率配置对任务掌握和通用知识保留的影响，并提供基于实证的重放比率选择指南。

### 方法

使用bAbI推理任务作为目标，应用合成数据生成方法，系统地评估不同的总令牌预算和重放比率配置，分析它们对任务掌握和通用知识保留的影响。

### 主要发现

实验揭示了一种最优配置，能够平衡特定任务性能与通用知识保留。基于研究发现，研究提供了基于计算预算选择重放比率的实证指导，使实践者能够在显著降低训练成本的情况下实现强大的任务适应。

### 结论

通过合理配置重放比率，实践者可以在有限的计算预算下实现有效的任务适应，同时保持模型的通用知识，从而显著降低训练成本。

### 翻译

通过持续预训练使语言模型适应新任务面临一个基本权衡：模型必须学习新能力，同时避免对现有知识的灾难性遗忘。虽然先前的工作已经研究了合成数据生成技术，但在计算约束下平衡任务性能和知识保留的最佳重放比率仍然知之甚少。我们提出了一个全面的实证研究，调查了在将语言模型适应新任务时，重放比率配置与计算预算之间的相互作用。使用bAbI推理任务作为我们的目标，我们应用合成数据生成方法，并系统评估不同的总令牌预算和重放比率配置。我们分析了它们对任务掌握和通用知识保留的影响。我们的实验揭示了一种平衡特定任务性能与通用知识保留的最优配置。基于我们的发现，我们提供了基于计算预算选择重放比率的实证指导，使实践者能够在显著降低训练成本的情况下实现强大的任务适应。


### 论文摘要

Adapting language models to new tasks through continued pretraining faces a fundamental trade-off: models must learn new capabilities while avoiding catastrophic forgetting of existing knowledge. While prior work has studied synthetic data generation techniques, the optimal replay ratios for balancing task performance and knowledge retention under computational constraints remain poorly understood. We present a comprehensive empirical study investigating the interplay between replay ratio configuration and computational budget when adapting language models to new tasks. Using the bAbI reasoning tasks as our target objective, we apply synthetic data generation and systematically evaluate different total token budgets and replay ratio configurations. We analyze their effects on both task mastery and general knowledge retention. Our experiments reveal an optimal configuration that balances task-specific performance with general knowledge retention. Based on our findings, we provide empirically-grounded guidelines for selecting replay ratios based on computational budget, enabling practitioners to achieve strong task adaptation with significantly reduced training costs.

---

## 34. Benchmarking foundation models for hyperspectral image classification: Application to cereal crop type mapping

**论文链接:** [http://arxiv.org/abs/2510.11576v2](http://arxiv.org/abs/2510.11576v2)

**作者:** Walid Elbarz, Mohamed Bourriz, Hicham Hajji, Hamd Ait Abdelali, François Bourzeix

**发布时间:** 2025-10-13

**备注:** currently being reviewed for WHISPERS conference ( Workshop on  Hyperspectral Image and Signal Processing: Evolution in Remote Sensing )

### GPT解析

### 总结

本研究评估了三种基础模型在高光谱作物制图中的性能，发现SpectralEarth预训练模型表现最佳，准确率达到93.5%，强调了模型架构在跨区域和传感器平台泛化能力中的重要性。

### 背景

基础模型正在改变地球观测领域，但它们在高光谱作物制图方面的潜力尚未被充分探索。

### 目的

对比评估三种基础模型（HyperSigma、DOFA和SpectralEarth数据集预训练的Vision Transformers）用于高光谱作物制图的性能。

### 方法

在训练区域的手动标记数据上对模型进行微调，在独立的测试区域评估模型性能，使用总体准确率、平均准确率和F1分数作为性能指标。

### 主要发现

HyperSigma的OA为34.5%（+/- 1.8%），DOFA达到62.6%（+/- 3.5%），SpectralEarth模型达到93.5%的OA（+/- 0.8%）；从头开始训练的紧凑型SpectralEarth变体达到91%的准确率，突显了模型架构对跨区域和传感器平台泛化能力的重要性。

### 结论

这些结果为基础模型用于实际高光谱作物制图提供了系统评估，为未来模型开发指明了方向。

### 翻译

基础模型正在改变地球观测，但它们在高光谱作物制图方面的潜力尚未被充分探索。本研究使用高光谱图像对三种基础模型（HyperSigma、DOFA和SpectralEarth数据集预训练的Vision Transformers）进行了基准测试，用于谷物作物制图。模型在训练区域的手动标记数据上进行了微调，并在独立的测试区域进行了评估。性能通过总体准确率、平均准确率和F1分数进行衡量。HyperSigma达到34.5%的OA（+/- 1.8%），DOFA达到62.6%（+/- 3.5%），SpectralEarth模型达到93.5%的OA（+/- 0.8%）。从零开始训练的紧凑型SpectralEarth变体达到91%，突显了模型架构对于在地理区域和传感器平台间实现强泛化能力的重要性。这些结果为基础模型用于实际高光谱作物制图提供了系统评估，并概述了未来模型开发的方向。


### 论文摘要

Foundation models are transforming Earth observation, but their potential for hyperspectral crop mapping remains underexplored. This study benchmarks three foundation models for cereal crop mapping using hyperspectral imagery: HyperSigma, DOFA, and Vision Transformers pre-trained on the SpectralEarth dataset (a large multitemporal hyperspectral archive). Models were fine-tuned on manually labeled data from a training region and evaluated on an independent test region. Performance was measured with overall accuracy (OA), average accuracy (AA), and F1-score. HyperSigma achieved an OA of 34.5% (+/- 1.8%), DOFA reached 62.6% (+/- 3.5%), and the SpectralEarth model achieved an OA of 93.5% (+/- 0.8%). A compact SpectralEarth variant trained from scratch achieved 91%, highlighting the importance of model architecture for strong generalization across geographic regions and sensor platforms. These results provide a systematic evaluation of foundation models for operational hyperspectral crop mapping and outline directions for future model development.

---

## 35. DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2510.12796v1](http://arxiv.org/abs/2510.12796v1)

**作者:** Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, Lu Hou, Lue Fan, Zhaoxiang Zhang

**发布时间:** 2025-10-14

### GPT解析

### 总结

DriveVLA-W0是一种通过世界模型预测未来图像的训练范式，解决了VLA模型的监督不足问题，显著提升了驾驶智能性能，并随着数据量增加性能提升加速。

### 背景

在大型数据上扩展视觉-语言-行动(VLA)模型是实现更通用驾驶智能的有前景路径，但VLA模型受到监督不足限制：模型容量大但仅由稀疏、低维度的行动监督，导致大部分表征能力未被利用。

### 目的

解决VLA模型的监督不足问题，使模型能够更好地学习驾驶环境的基本动态，提高驾驶智能的泛化能力。

### 方法

提出DriveVLA-W0训练范式，采用世界模型预测未来图像生成密集自监督信号；为两种VLA架构实现：离散视觉令牌的自回归世界模型和连续视觉特征的扩散世界模型；引入轻量级行动专家解决实时部署推理延迟问题。

### 主要发现

在NAVSIM v1/v2基准测试和680倍大的内部数据集上，DriveVLA-W0显著优于BEV和VLA基线；放大了数据扩展定律，表明随着训练数据集大小增加，性能提升加速。

### 结论

DriveVLA-W0通过世界模型生成密集自监督信号有效解决VLA模型监督不足问题，提高模型表征能力同时通过轻量级行动专家解决实时部署问题，具有良好的可扩展性。

### 翻译

在大型数据上扩展视觉-语言-行动(VLA)模型是实现更通用驾驶智能的有前景路径。然而，VLA模型受到监督不足的限制：模型容量巨大但仅由稀疏、低维度的行动监督，导致其大部分表征能力未被充分利用。为解决此问题，我们提出DriveVLA-W0训练范式，采用世界模型来预测未来图像。此任务生成密集的自监督信号，迫使模型学习驾驶环境的基本动态。我们通过为两种主导的VLA架构实现该范式来展示其多功能性：用于使用离散视觉令牌的VLA的自回归世界模型，以及用于在连续视觉特征上操作的VLA的扩散世界模型。基于从世界模型学到的丰富表征，我们引入轻量级行动专家以解决实时部署的推理延迟问题。在NAVSIM v1/v2基准测试和680倍大的内部数据集上进行的大量实验表明，DriveVLA-W0显著优于BEV和VLA基线。关键是，它放大了数据扩展定律，表明随着训练数据集大小的增加，性能提升会加速。


### 论文摘要

Scaling Vision-Language-Action (VLA) models on large-scale data offers a promising path to achieving a more generalized driving intelligence. However, VLA models are limited by a ``supervision deficit'': the vast model capacity is supervised by sparse, low-dimensional actions, leaving much of their representational power underutilized. To remedy this, we propose \textbf{DriveVLA-W0}, a training paradigm that employs world modeling to predict future images. This task generates a dense, self-supervised signal that compels the model to learn the underlying dynamics of the driving environment. We showcase the paradigm's versatility by instantiating it for two dominant VLA archetypes: an autoregressive world model for VLAs that use discrete visual tokens, and a diffusion world model for those operating on continuous visual features. Building on the rich representations learned from world modeling, we introduce a lightweight action expert to address the inference latency for real-time deployment. Extensive experiments on the NAVSIM v1/v2 benchmark and a 680x larger in-house dataset demonstrate that DriveVLA-W0 significantly outperforms BEV and VLA baselines. Crucially, it amplifies the data scaling law, showing that performance gains accelerate as the training dataset size increases.

---

## 36. Towards Fast Coarse-graining and Equation Discovery with Foundation Inference Models

**论文链接:** [http://arxiv.org/abs/2510.12618v1](http://arxiv.org/abs/2510.12618v1)

**作者:** Manuel Hinz, Maximilian Mauel, Patrick Seifner, David Berghaus, Kostadin Cvejoski, Ramses J. Sanchez

**发布时间:** 2025-10-14

### GPT解析

### 总结

本文提出了一种解耦方法，利用基础推理模型(FIMs)来识别高维动态系统中的潜在变量，通过冻结FIM权重并只训练编码器-解码器映射，实现稳定高效的表示学习。

### 背景

高维动态过程通常由更小的有效变量集合表征，这些变量在低维流形上演化。识别这些潜在动态需要解决两个交织的问题：发现适当的粗粒度变量和同时拟合控制方程。

### 目的

将变量发现和动态拟合两个问题解耦，利用预训练的基础推理模型(FIMs)简化高维动态系统的分析过程。

### 方法

通过利用预训练的基础推理模型(FIMs)来估计动态系统的无穷小生成器，使用具有冻结权重的FIM来推断动态，同时只训练编码器-解码器映射，定义了一个简单、模拟一致的损失函数来稳定表示学习。

### 主要发现

在具有半圆扩散的随机双阱系统（嵌入到合成视频数据中）的概念验证实验表明，该方法具有快速和可重用的粗粒度处理流程的潜力。

### 结论

通过解耦变量发现和动态拟合的问题，并利用预训练的FIMs，提出了一种稳定且高效的表示学习方法，适用于高维动态系统的粗粒度化。

### 翻译

高维动态过程的记录通常由更小的有效变量集合表征，这些变量在低维流形上演化。识别这些潜在动态需要解决两个交织的问题：发现适当的粗粒度变量和同时拟合控制方程。大多数机器学习方法通过联合训练自动编码器和强制动态一致性的模型来解决这些任务。我们提出通过利用最近引入的基础推理模型(FIMs)来解耦这两个问题。FIMs是预训练模型，可以零样本模式估计动态系统的无穷小生成器（例如随机微分方程的漂移和扩散）。通过使用具有冻结权重的FIM来推断动态，并且只训练编码器-解码器映射，我们定义了一个简单、模拟一致的损失函数，稳定了表示学习。在一个具有半圆扩散的随机双阱系统上进行的嵌入合成视频数据的概念证明，展示了这种方法在快速和可重用的粗粒度处理流程方面的潜力。


### 论文摘要

High-dimensional recordings of dynamical processes are often characterized by a much smaller set of effective variables, evolving on low-dimensional manifolds. Identifying these latent dynamics requires solving two intertwined problems: discovering appropriate coarse-grained variables and simultaneously fitting the governing equations. Most machine learning approaches tackle these tasks jointly by training autoencoders together with models that enforce dynamical consistency. We propose to decouple the two problems by leveraging the recently introduced Foundation Inference Models (FIMs). FIMs are pretrained models that estimate the infinitesimal generators of dynamical systems (e.g., the drift and diffusion of a stochastic differential equation) in zero-shot mode. By amortizing the inference of the dynamics through a FIM with frozen weights, and training only the encoder-decoder map, we define a simple, simulation-consistent loss that stabilizes representation learning. A proof of concept on a stochastic double-well system with semicircle diffusion, embedded into synthetic video data, illustrates the potential of this approach for fast and reusable coarse-graining pipelines.

---

## 37. LayerSync: Self-aligning Intermediate Layers

**论文链接:** [http://arxiv.org/abs/2510.12581v1](http://arxiv.org/abs/2510.12581v1)

**作者:** Yasaman Haghighi, Bastien van Delft, Mariam Hassan, Alexandre Alahi

**发布时间:** 2025-10-14

### GPT解析

### 总结

LayerSync是一种领域无关的方法，用于提高扩散模型的生成质量和训练效率。它通过利用模型自身的中间表示来正则化模型，减少对外部监督的需求，是一种自给自足、即插即用的正则化项，不需要预训练模型或额外数据，可应用于多种模态。

### 背景

先前的研究已经指出扩散模型的生成质量与模型学习的表示之间存在联系，表明对模型中间表示的外部指导可以加速训练。然而，扩散模型不同层的表示质量存在差异。

### 目的

重新构想扩散模型的训练范式，通过利用模型自身的中间表示来正则化模型，从而减少对外部监督的需求，提高生成质量和训练效率。

### 方法

LayerSync基于扩散模型不同层表示质量不同的观察，将语义最丰富的表示作为较弱表示的内在指导。这是一种自给自足、即插即用的正则化项，不需要在扩散模型训练中增加额外开销，可以推广到视觉领域以外的其他模态。

### 主要发现

LayerSync不需要预训练模型或额外数据；在图像生成上进行了广泛评估，并展示了其在音频、视频和运动生成等其他领域的适用性；它持续提高了生成质量和训练效率；例如，在ImageNet数据集上将基于流的变压器的训练速度提高了8.75倍以上，并将生成质量提高了23.6%。

### 结论

LayerSync是一种有效的方法，可以提高扩散模型的生成质量和训练效率，适用于多种模态，且不需要额外的计算资源或数据。

### 翻译

我们提出了LayerSync，一种领域无关的方法，用于提高扩散模型的生成质量和训练效率。先前的研究已经强调了扩散模型生成的质量与模型学习的表示之间的联系，表明对模型中间表示的外部指导可以加速训练。我们通过用模型的自身中间表示来正则化扩散模型，重新构想了这一范式。基于扩散模型不同层的表示质量存在差异的观察，我们证明语义最丰富的表示可以作为较弱表示的内在指导，减少对外部监督的需求。我们的方法LayerSync是一种自给自足、即插即用的正则化项，在扩散模型训练中没有额外开销，并且可以推广到视觉领域以外的其他模态。LayerSync不需要预训练模型或额外数据。我们在图像生成上广泛评估了该方法，并展示了其在音频、视频和运动生成等其他领域的适用性。我们表明它持续提高了生成质量和训练效率。例如，我们在ImageNet数据集上将基于流变压器的训练速度提高了8.75倍以上，并将生成质量提高了23.6%。代码可在https://github.com/vita-epfl/LayerSync获取。


### 论文摘要

We propose LayerSync, a domain-agnostic approach for improving the generation quality and the training efficiency of diffusion models. Prior studies have highlighted the connection between the quality of generation and the representations learned by diffusion models, showing that external guidance on model intermediate representations accelerates training. We reconceptualize this paradigm by regularizing diffusion models with their own intermediate representations. Building on the observation that representation quality varies across diffusion model layers, we show that the most semantically rich representations can act as an intrinsic guidance for weaker ones, reducing the need for external supervision. Our approach, LayerSync, is a self-sufficient, plug-and-play regularizer term with no overhead on diffusion model training and generalizes beyond the visual domain to other modalities. LayerSync requires no pretrained models nor additional data. We extensively evaluate the method on image generation and demonstrate its applicability to other domains such as audio, video, and motion generation. We show that it consistently improves the generation quality and the training efficiency. For example, we speed up the training of flow-based transformer by over 8.75x on ImageNet dataset and improved the generation quality by 23.6%. The code is available at https://github.com/vita-epfl/LayerSync.

---

## 38. I-DCCRN-VAE: An Improved Deep Representation Learning Framework for Complex VAE-based Single-channel Speech Enhancement

**论文链接:** [http://arxiv.org/abs/2510.12485v1](http://arxiv.org/abs/2510.12485v1)

**作者:** Jiatong Li, Simon Doclo

**发布时间:** 2025-10-14

### GPT解析

### 总结

本文提出了一种改进的DCCRN-VAE单通道语音增强系统，通过移除预训练VAE中的跳跃连接、使用β-VAE进行预训练、以及让NSVAE同时生成语音和噪声潜在表示，提高了系统在不匹配数据集上的泛化能力。

### 背景

最近提出了一种基于复杂变分自编码器(VAE)的单通道语音增强系统，该系统基于DCCRN架构。在这个系统中，噪声抑制VAE(NSVAE)使用预训练的干净语音和噪声VAE以及跳跃连接来从嘈杂语音中提取干净语音表示。

### 目的

改进DCCRN-VAE系统，提高其在不同数据集上的泛化能力。

### 方法

通过三个关键改进：1) 移除预训练VAE中的跳跃连接，以鼓励更具信息性的语音和噪声潜在表示；2) 在预训练中使用β-VAE，以更好地平衡重建和潜在空间正则化；3) NSVAE同时生成语音和噪声潜在表示。

### 主要发现

在匹配的DNS3数据集上，所提出的系统与DCCRN和DCCRN-VAE基线实现了相当的性能；在不匹配的数据集(WSJ0-QUT, Voicebank-DEMEND)上，优于基线，显示出改进的泛化能力；消融研究表明，可以使用传统的微调而非对抗训练实现类似性能，从而简化训练流程。

### 结论

所提出的改进提高了系统在不匹配数据集上的泛化能力；简化的训练流程(使用传统微调而非对抗训练)也能获得类似性能。

### 翻译

最近，提出了一种基于复杂变分自编码器(VAE)的基于DCCRN架构的单通道语音增强系统。在该系统中，噪声抑制VAE(NSVAE)使用预训练的干净语音和噪声VAE以及跳跃连接，从嘈杂语音中学习提取干净语音表示。在本文中，我们通过三个关键改进来改进DCCRN-VAE：1) 移除预训练VAE中的跳跃连接，以鼓励更具信息性的语音和噪声潜在表示；2) 在预训练中使用β-VAE，以更好地平衡重建和潜在空间正则化；3) NSVAE生成语音和噪声潜在表示。实验表明，所提出的系统在匹配的DNS3数据集上实现了与DCCRN和DCCRN-VAE基线相当的性能，但在不匹配的数据集(WSJ0-QUT, Voicebank-DEMEND)上优于基线，显示出改进的泛化能力。此外，消融研究表明，可以使用传统的微调而非对抗训练实现类似性能，从而简化训练流程。


### 论文摘要

Recently, a complex variational autoencoder (VAE)-based single-channel speech enhancement system based on the DCCRN architecture has been proposed. In this system, a noise suppression VAE (NSVAE) learns to extract clean speech representations from noisy speech using pretrained clean speech and noise VAEs with skip connections. In this paper, we improve DCCRN-VAE by incorporating three key modifications: 1) removing the skip connections in the pretrained VAEs to encourage more informative speech and noise latent representations; 2) using $\beta$-VAE in pretraining to better balance reconstruction and latent space regularization; and 3) a NSVAE generating both speech and noise latent representations. Experiments show that the proposed system achieves comparable performance as the DCCRN and DCCRN-VAE baselines on the matched DNS3 dataset but outperforms the baselines on mismatched datasets (WSJ0-QUT, Voicebank-DEMEND), demonstrating improved generalization ability. In addition, an ablation study shows that a similar performance can be achieved with classical fine-tuning instead of adversarial training, resulting in a simpler training pipeline.

---

## 39. SMEC: Rethinking Matryoshka Representation Learning for Retrieval Embedding Compression

**论文链接:** [http://arxiv.org/abs/2510.12474v1](http://arxiv.org/abs/2510.12474v1)

**作者:** Biao Zhang, Lixin Chen, Tong Liu, Bo Zheng

**发布时间:** 2025-10-14

**备注:** Accepted by EMNLP2025

### GPT解析

### 总结

本文提出了一种名为SMEC的新型训练框架，用于压缩大型语言模型的高维嵌入向量，解决了高维度带来的计算复杂度和存储需求问题。

### 背景

大型语言模型生成的高维嵌入向量虽然能捕捉丰富的语义和句法信息，但高维度加剧了计算复杂度和存储需求，阻碍了实际部署。

### 目的

解决高维嵌入带来的计算复杂度和存储需求问题，实现有效的维度压缩而不损失性能。

### 方法

提出Sequential Matryoshka Embedding Compression (SMEC)框架，包含三个核心组件：Sequential Matryoshka Representation Learning (SMRL)方法减轻训练中的梯度方差，Adaptive Dimension Selection (ADS)模块减少维度修剪时的信息损失，Selectable Cross-batch Memory (S-XBM)模块增强高维和低维嵌入间的无监督学习。

### 主要发现

在图像、文本和多模态数据集上的实验表明，SMEC能在显著降低维度的同时保持性能。在BEIR数据集上，与Matryoshka-Adaptor和Search-Adaptor模型相比，SMEC将压缩后的LLM2Vec嵌入向量(256维)的性能分别提高了1.1分和2.7分。

### 结论

SMEC框架能够有效压缩大型语言模型的高维嵌入向量，在减少计算复杂度和存储需求的同时保持或提升模型性能。

### 翻译

大型语言模型生成的高维嵌入向量能够捕捉丰富的语义和句法信息。然而，高维嵌入向量加剧了计算复杂度和存储需求，从而阻碍了实际部署。为解决这些挑战，我们提出了一种名为Sequential Matryoshka Embedding Compression (SMEC)的新型训练框架。该框架引入了Sequential Matryoshka Representation Learning (SMRL)方法来减轻训练过程中的梯度方差，Adaptive Dimension Selection (ADS)模块来减少维度修剪过程中的信息损失，以及Selectable Cross-batch Memory (S-XBM)模块来增强高维和低维嵌入之间的无监督学习。在图像、文本和多模态数据集上的实验表明，SMEC在显著降低维度的同时保持了性能。例如，在BEIR数据集上，与Matryoshka-Adaptor和Search-Adaptor模型相比，我们的方法将压缩后的LLM2Vec嵌入向量(256维)的性能分别提高了1.1分和2.7分。


### 论文摘要

Large language models (LLMs) generate high-dimensional embeddings that capture rich semantic and syntactic information. However, high-dimensional embeddings exacerbate computational complexity and storage requirements, thereby hindering practical deployment. To address these challenges, we propose a novel training framework named Sequential Matryoshka Embedding Compression (SMEC). This framework introduces the Sequential Matryoshka Representation Learning(SMRL) method to mitigate gradient variance during training, the Adaptive Dimension Selection (ADS) module to reduce information degradation during dimension pruning, and the Selectable Cross-batch Memory (S-XBM) module to enhance unsupervised learning between high- and low-dimensional embeddings. Experiments on image, text, and multimodal datasets demonstrate that SMEC achieves significant dimensionality reduction while maintaining performance. For instance, on the BEIR dataset, our approach improves the performance of compressed LLM2Vec embeddings (256 dimensions) by 1.1 points and 2.7 points compared to the Matryoshka-Adaptor and Search-Adaptor models, respectively.

---

## 40. Deep SPI: Safe Policy Improvement via World Models

**论文链接:** [http://arxiv.org/abs/2510.12312v1](http://arxiv.org/abs/2510.12312v1)

**作者:** Florent Delgrange, Raphael Avalos, Willem Röpke

**发布时间:** 2025-10-14

**备注:** 10 pages main text, 17 pages appendix (excluding references)

### GPT解析

### 总结

研究在线强化学习环境下的安全策略改进(SPI)理论，结合世界模型和表示学习，提出DeepSPI算法，在保持理论保证的同时实现了优异性能。

### 背景

现有SPI保证主要关注离线、表格强化学习环境，缺乏在线设置下的理论支持。

### 目的

开发理论框架，限制策略更新到当前策略的明确定义邻域，确保策略单调改进和收敛。

### 方法

分析转换和奖励预测损失与表示质量的关系，开发在线、深度版本的经典SPI定理，提出DeepSPI算法，结合局部转换和奖励损失与正则化策略更新。

### 主要发现

限制策略更新到当前策略的邻域可以确保单调改进和收敛；转换和奖励预测损失与表示质量相关联；DeepSPI在ALE-57基准测试中匹配或超过PPO和DeepMDPs等强基线方法。

### 结论

DeepSPI算法在保持理论保证的同时，在实际应用中展现出与最先进方法相当或更好的性能。

### 翻译

安全策略改进(SPI)为策略更新提供了理论控制，但现有保证主要涉及离线、表格强化学习(RL)。我们研究结合世界模型和表示学习的在线设置下的SPI。我们开发了一个理论框架，显示将策略更新限制在当前策略的明确定义邻域内可以确保单调改进和收敛。该分析将转换和奖励预测损失与表示质量联系起来，产生了来自离线RL文献的经典SPI定理的在线、'深度'类比。基于这些结果，我们引入了DeepSPI，一种将局部转换和奖励损失与正则化策略更新相结合的原则性在线策略算法。在ALE-57基准测试中，DeepSPI匹配或超过了包括PPO和DeepMDPs在内的强基线方法，同时保留了理论保证。


### 论文摘要

Safe policy improvement (SPI) offers theoretical control over policy updates, yet existing guarantees largely concern offline, tabular reinforcement learning (RL). We study SPI in general online settings, when combined with world model and representation learning. We develop a theoretical framework showing that restricting policy updates to a well-defined neighborhood of the current policy ensures monotonic improvement and convergence. This analysis links transition and reward prediction losses to representation quality, yielding online, "deep" analogues of classical SPI theorems from the offline RL literature. Building on these results, we introduce DeepSPI, a principled on-policy algorithm that couples local transition and reward losses with regularised policy updates. On the ALE-57 benchmark, DeepSPI matches or exceeds strong baselines, including PPO and DeepMDPs, while retaining theoretical guarantees.

---

## 41. Unveiling the Vulnerability of Graph-LLMs: An Interpretable Multi-Dimensional Adversarial Attack on TAGs

**论文链接:** [http://arxiv.org/abs/2510.12233v1](http://arxiv.org/abs/2510.12233v1)

**作者:** Bowen Fan, Zhilin Guo, Xunkai Li, Yihan Zhou, Bing Zhou, Zhenjun Li, Rong-Hua Li, Guoren Wang

**发布时间:** 2025-10-14

**备注:** 12 pages, 4 figures

### GPT解析

### 总结

本研究提出了IMDGA框架，一种统一的多维图攻击方法，针对图神经网络与大型语言模型结合的文本属性图，通过协调图结构和文本特征的多层次扰动，实现对Graph-LLMs的有效攻击，同时保持高度可解释性。

### 背景

图神经网络已成为建模图结构数据的关键框架，通过整合大型语言模型，文本属性图利用丰富的文本语义显著提高了图学习能力。然而，这种协同作用也引入了关键漏洞，使Graph-LLMs容易受到对其结构拓扑和文本属性的对抗攻击。

### 目的

虽然已有针对结构拓扑和文本属性的专门攻击方法，但缺乏统一的多维攻击框架。本研究旨在提出一种同时考虑图结构和文本特征的对抗攻击方法，并平衡攻击效果与可解释性。

### 方法

作者提出了可解释的多维图攻击（IMDGA）框架，该框架设计用于协调图结构和文本特征的多层次扰动。IMDGA利用三个紧密集成的模块构建攻击，平衡可解释性和影响力，帮助更深入理解Graph-LLM的漏洞。

### 主要发现

通过在不同数据集和架构上的理论分析和实证评估，IMDGA在可解释性、攻击有效性、隐蔽性和鲁棒性方面均优于现有方法。研究揭示了TAG表示学习中的关键弱点，发现了Graph-LLMs中一个先前未被充分探索的语义维度漏洞。

### 结论

通过暴露Graph-LLMs中的关键弱点，这项工作为提高系统弹性提供了有价值的见解，相关代码和资源已公开分享。

### 翻译

图神经网络已成为建模图结构数据的关键框架，应用于从社交网络分析到分子化学的广泛领域。通过整合大型语言模型，文本属性图利用丰富的文本语义增强节点表示，显著提高了基于图的学习能力。然而，这种复杂的协同作用引入了关键漏洞，因为图-LLMs容易受到对其结构拓扑和文本属性的对抗攻击。虽然已经为这些方面的每个方面设计了专门的攻击方法，但还没有将它们统一为全面的方法。在本工作中，我们提出了可解释的多维图攻击（IMDGA），这是一种新型的人本主义对抗攻击框架，旨在协调图结构和文本特征的多层次扰动。IMDGA利用三个紧密集成的模块来构建攻击，平衡可解释性和影响力，使人们能够更深入地理解Graph-LLM的漏洞。通过在不同数据集和架构上进行严格的理论分析和全面的实证评估，IMDGA显示出比现有方法更好的可解释性、攻击有效性、隐蔽性和鲁棒性。通过揭示TAG表示学习中的关键弱点，这项工作揭示了Graph-LLMs中一个先前未被充分探索的语义维度漏洞，为提高它们的弹性提供了有价值的见解。我们的代码和资源已在https://anonymous.4open.science/r/IMDGA-7289公开提供。


### 论文摘要

Graph Neural Networks (GNNs) have become a pivotal framework for modeling graph-structured data, enabling a wide range of applications from social network analysis to molecular chemistry. By integrating large language models (LLMs), text-attributed graphs (TAGs) enhance node representations with rich textual semantics, significantly boosting the expressive power of graph-based learning. However, this sophisticated synergy introduces critical vulnerabilities, as Graph-LLMs are susceptible to adversarial attacks on both their structural topology and textual attributes. Although specialized attack methods have been designed for each of these aspects, no work has yet unified them into a comprehensive approach. In this work, we propose the Interpretable Multi-Dimensional Graph Attack (IMDGA), a novel human-centric adversarial attack framework designed to orchestrate multi-level perturbations across both graph structure and textual features. IMDGA utilizes three tightly integrated modules to craft attacks that balance interpretability and impact, enabling a deeper understanding of Graph-LLM vulnerabilities. Through rigorous theoretical analysis and comprehensive empirical evaluations on diverse datasets and architectures, IMDGA demonstrates superior interpretability, attack effectiveness, stealthiness, and robustness compared to existing methods. By exposing critical weaknesses in TAG representation learning, this work uncovers a previously underexplored semantic dimension of vulnerability in Graph-LLMs, offering valuable insights for improving their resilience. Our code and resources are publicly available at https://anonymous.4open.science/r/IMDGA-7289.

---

## 42. DE3S: Dual-Enhanced Soft-Sparse-Shape Learning for Medical Early Time-Series Classification

**论文链接:** [http://arxiv.org/abs/2510.12214v1](http://arxiv.org/abs/2510.12214v1)

**作者:** Tao Xie, Zexi Tan, Haoyi Xiao, Binbin Sun, Yiqun Zhang

**发布时间:** 2025-10-14

**备注:** Accepted to IEEE BIBM 2025

### GPT解析

### 总结

本文提出了一种名为DE3S的医疗早期时间序列分类方法，通过双增强软稀疏形状学习解决了医疗场景中早期分类面临的准确性和早期性权衡问题。

### 背景

早期时间序列分类在医疗应用中至关重要，特别是在ICU败血症预测等时间敏感场景中，延迟预测会导致大量死亡。ETSC可提高ICU资源利用效率和医疗精准度，但面临初始信号弱和类别不平衡的挑战。

### 目的

解决ETSC中准确性和早期性的冲突目标，捕捉早期细微模式，找到具有高可解释性的区分性子序列（形状）。

### 方法

提出DE3S方法，包含三个创新：1)结合传统时间增强和基于注意力的全局时间增强的双增强策略；2)基于注意力分数的软形状稀疏化机制；3)双路径MoE和Inception模块融合架构。使用加权交叉熵损失处理类别不平衡。

### 主要发现

在六个真实医疗数据集上进行了广泛实验，结果显示了最先进的性能，消融研究证实了各组件的有效性。

### 结论

DE3S方法成功解决了医疗早期时间序列分类中的准确性和早期性权衡问题，能够有效捕捉早期细微模式，提高ICU资源利用效率和医疗精准度。

### 翻译

医疗应用中的早期时间序列分类(ETSC)对于ICU中败血症预测等时间敏感场景至关重要，大量死亡是由延迟预测引起的。ETSC可以显著提高ICU资源利用效率和医疗精准度。然而，它面临准确性和早期性的冲突目标，现有方法常常在两者之间权衡，由于初始信号弱和类别不平衡，难以捕捉早期的细微模式。解决这些挑战的关键是找到具有高可解释性的区分性子序列（或形状）。本文提出了用于医疗早期时间序列分类的双增强软稀疏形状学习(DE3S)，它引入了一种新的双增强软形状学习框架，通过三个创新精确找出形状：1)结合传统时间增强和基于注意力的全局时间增强的全面双增强策略，实现鲁棒的表示学习；2)基于注意力分数的软形状稀疏化机制，动态保留区分性模式，同时将不太重要的形状聚合成代表性标记；3)双路径专家混合网络(MoE)和Inception模块融合架构，其中MoE在形状内执行局部学习，多尺度Inception模块跨形状捕获全局模式。该框架使用加权交叉熵损失处理类别不平衡，并在主体一致性数据集上表现出鲁棒性。在六个真实医疗数据集上的广泛实验显示了最先进的性能，消融研究证实了组件的有效性。


### 论文摘要

Early time-series classification (ETSC) in medical applications is crucial for time-sensitive scenarios such as sepsis prediction in intensive care units (ICUs), where a large number of deaths are caused by delayed prediction. ETSC can significantly improve ICU resource utilization efficiency and healthcare precision. However, it faces conflicting goals of accuracy and earliness, with existing methods often trading one for the other, struggling to capture subtle early-stage patterns due to weak initial signals and class imbalance. The key to solve these challenges is to find shapelets, which are discriminative subsequences (or shapes) with high interpretability in time-series classification. This paper proposes Dual-Enhanced Soft-Sparse-Shape Learning for Medical Early Time-Series Classification (DE3S), which introduces a novel Dual-Enhanced Soft-Shape Learning framework to figure out shapelets precisely through three innovations: (1) a comprehensive dual-enhancement strategy combines traditional temporal augmentation with attention-based global temporal enhancement for robust representation learning, (2) an attention-score-based soft shapelet sparsification mechanism dynamically preserves discriminative patterns while aggregating less important shapelets into representative tokens, and (3) a dual-path Mixture of Experts Network (MoE) and Inception modules fusion architecture where MoE performs local learning within shapelets and multi-scale Inception modules capture global patterns across shapelets. The framework employs weighted cross-entropy loss for class imbalance handling and demonstrates robustness on subject-consistency datasets. Extensive experiments on six real-world medical datasets show state-of-the-art performance, with ablation studies confirming component efficacy.

---

## 43. SDGraph: Multi-Level Sketch Representation Learning by Sparse-Dense Graph Architecture

**论文链接:** [http://arxiv.org/abs/2510.12192v1](http://arxiv.org/abs/2510.12192v1)

**作者:** Xi Cheng, Pingfa Feng, Zhichao Liao, Mingyu Fan, Long Zeng

**发布时间:** 2025-10-14

### GPT解析

### 总结

该研究针对手绘草图的独特稀疏性和抽象性，提出了多级草图表示方案和SDGraph深度学习架构，有效利用草图中的有效信息，在多个下游任务中取得了优于现有方法的性能。

### 背景

手绘草图具有独特的稀疏性和抽象性，需要与图像不同的学习流程。然而，对于什么是有效的草图信息的研究有限，这限制了现有方法的性能。

### 目的

系统地识别和利用草图中的有效信息，以提升草图学习方法的性能。

### 方法

提出多级草图表示方案，将草图表示分为草图级、笔画级和点级三个层次；基于此开发了SDGraph深度学习架构，包含稀疏图和密集图两个互补模块，以及信息融合模块。

### 主要发现

通过理论分析和实验评估，确定了草图中的有效信息；SDGraph在分类、检索和矢量草图生成任务上分别比最先进方法提高了1.15%、1.70%和36.58%的性能。

### 结论

多级草图表示方案能够系统地识别有效信息，SDGraph架构能够有效利用这些信息，提升各种草图相关任务的性能。

### 翻译

手绘草图具有独特的稀疏性和抽象性，需要不同于图像的学习流程。对于草图学习方法，主要目标是充分利用草图中的有效信息。然而，关于什么是有效的草图信息的研究有限，这限制了现有方法的性能。为解决这一问题，我们首先提出了多级草图表示方案，系统地识别有效信息。该方案将草图表示分为三个层次：草图级、笔画级和点级。此设计基于分析元素的粒度，从粗（草图级）到细（点级），从而确保更全面地覆盖草图信息。对每个层次，我们进行了理论分析和实验评估，以识别和验证有效信息。基于上述研究，我们开发了SDGraph，这是一个深度学习架构，旨在利用三个层次中识别出的有效信息。SDGraph包含两个互补模块：稀疏图将笔画作为节点，用于草图级和笔画级表示学习；密集图将点作为节点，用于草图级和点级表示学习。两个模块都采用图卷积以及下采样和上采样操作，使其能够作为编码器和解码器。此外，信息融合模块连接两个图，进一步增强特征提取。SDGraph支持各种草图相关的下游任务，在分类和检索方面分别比最先进方法提高了1.15%和1.70%，在矢量草图生成质量上提高了36.58%。


### 论文摘要

Freehand sketches exhibit unique sparsity and abstraction, necessitating learning pipelines distinct from those designed for images. For sketch learning methods, the central objective is to fully exploit the effective information embedded in sketches. However, there is limited research on what constitutes effective sketch information, which in turn constrains the performance of existing approaches. To tackle this issue, we first proposed the Multi-Level Sketch Representation Scheme to systematically identify the effective information. The scheme organizes sketch representation into three levels: sketch-level, stroke-level, and point-level. This design is based on the granularity of analytical elements, from coarse (sketch-level) to fine (point-level), thereby ensuring more comprehensive coverage of the sketch information. For each level, we conducted theoretical analyses and experimental evaluations to identify and validate the effective information. Building on the above studies, we developed SDGraph, a deep learning architecture designed to exploit the identified effective information across the three levels. SDGraph comprises two complementary modules: a Sparse Graph that treats strokes as nodes for sketch-level and stroke-level representation learning, and a Dense Graph that treats points as nodes for sketch-level and point-level representation learning. Both modules employ graph convolution along with down-sampling and up-sampling operations, enabling them to function as both encoder and decoder. Besides that, an information fusion module bridges the two graphs to further enhance feature extraction. SDGraph supports a wide range of sketch-related downstream tasks, achieving accuracy improvements of 1.15\% and 1.70\% over the state-of-the-art in classification and retrieval, respectively, and 36.58\% improvement in vector sketch generation quality.

---

## 44. DRL: Discriminative Representation Learning with Parallel Adapters for Class Incremental Learning

**论文链接:** [http://arxiv.org/abs/2510.12107v1](http://arxiv.org/abs/2510.12107v1)

**作者:** Jiawei Zhan, Jun Liu, Jinlong Peng, Xiaochen Chen, Bin-Bin Gao, Yong Liu, Chengjie Wang

**发布时间:** 2025-10-14

**备注:** 13 pages, 7 figures

### GPT解析

### 总结

本研究提出了区分性表示学习(DRL)框架，通过增量并行适配器网络和解耦锚监督方法，解决了非重排类增量学习中的三大挑战，在保持高效率的同时显著提升了性能。

### 背景

预训练模型(PTMs)在非重排类增量学习(CIL)研究中表现出色，但仍面临三大挑战：模型复杂度不断增加、增量学习过程中表示不平稳、以及阶段性子问题优化与全局推理之间不一致。

### 目的

设计一个能够有效解决非重排类增量学习中的三大挑战的框架，实现平稳的表示转移，并缩小阶段性局部优化与全局推理之间的差距。

### 方法

提出区分性表示学习(DRL)框架，包含增量并行适配器(IPA)网络和解耦锚监督(DAS)机制。IPA网络基于预训练模型构建，通过学习轻量级适配器在每个增量阶段逐步增强模型；DAS机制通过分别比较正负样本与虚拟锚来解耦约束，促进区分性表示学习并实现对齐不同阶段特征空间。

### 主要发现

在六个基准测试上的实验表明，DRL在整个CIL期间持续优于其他最先进方法，同时在训练和推理阶段都保持高效率，有效解决了模型复杂度、表示不平稳和优化不一致问题。

### 结论

DRL框架通过创新性的网络架构和监督机制，成功解决了非重排类增量学习中的关键挑战，为该领域提供了高效且有效的解决方案。

### 翻译

凭借预训练模型(PTMs)的优秀表示能力，非重排类增量学习(CIL)研究取得了显著进展。然而，由于三个难题：日益增长的大模型复杂度、增量学习过程中不平稳的表示转移、以及阶段性子问题优化与全局推理之间的不一致性，这仍然是一个极具挑战性的任务。在这项工作中，我们提出了区分性表示学习(DRL)框架来专门解决这些挑战。为了有效且高效地进行增量学习，DRL的网络称为增量并行适配器(IPA)网络，它基于PTM构建，并通过在每个增量阶段学习轻量级适配器来逐步增强模型，参数学习开销小。该适配器负责使模型适应新类别，它可以通过它们之间的并行连接和传输门继承并传播当前模型的表示能力。因此，这种设计保证了不同增量阶段之间的平稳表示转移。此外，为了缓解不一致性并实现跨增量阶段可比的特征表示，我们设计了解耦锚监督(DAS)。它通过将正样本和负样本分别与虚拟锚进行比较来解耦它们的约束。这种解耦促进了区分性表示学习并对齐了在不同阶段学习的特征空间，从而缩小了在数据子集上进行阶段性局部优化与在所有类别上进行全局推理之间的差距。在六个基准测试上的大量实验表明，我们的DRL在整个CIL期间持续优于其他最先进方法，同时在训练和推理阶段都保持高效率。


### 论文摘要

With the excellent representation capabilities of Pre-Trained Models (PTMs), remarkable progress has been made in non-rehearsal Class-Incremental Learning (CIL) research. However, it remains an extremely challenging task due to three conundrums: increasingly large model complexity, non-smooth representation shift during incremental learning and inconsistency between stage-wise sub-problem optimization and global inference. In this work, we propose the Discriminative Representation Learning (DRL) framework to specifically address these challenges. To conduct incremental learning effectively and yet efficiently, the DRL's network, called Incremental Parallel Adapter (IPA) network, is built upon a PTM and increasingly augments the model by learning a lightweight adapter with a small amount of parameter learning overhead in each incremental stage. The adapter is responsible for adapting the model to new classes, it can inherit and propagate the representation capability from the current model through parallel connection between them by a transfer gate. As a result, this design guarantees a smooth representation shift between different incremental stages. Furthermore, to alleviate inconsistency and enable comparable feature representations across incremental stages, we design the Decoupled Anchor Supervision (DAS). It decouples constraints of positive and negative samples by respectively comparing them with the virtual anchor. This decoupling promotes discriminative representation learning and aligns the feature spaces learned at different stages, thereby narrowing the gap between stage-wise local optimization over a subset of data and global inference across all classes. Extensive experiments on six benchmarks reveal that our DRL consistently outperforms other state-of-the-art methods throughout the entire CIL period while maintaining high efficiency in both training and inference phases.

---

## 45. MIARec: Mutual-influence-aware Heterogeneous Network Embedding for Scientific Paper Recommendation

**论文链接:** [http://arxiv.org/abs/2510.12054v1](http://arxiv.org/abs/2510.12054v1)

**作者:** Wenjin Xie, Tao Jia

**发布时间:** 2025-10-14

### GPT解析

### 总结

该论文提出了一种名为MIARec的学术论文推荐模型，通过基于引力的方法衡量学者之间的相互学术影响，并将其整合到图表示学习的特征聚合过程中，以解决传统基于图的推荐方法忽视学术网络中不对称学术影响的问题。

### 背景

随着科学文献的快速扩张，学者们对精确且高质量的论文推荐需求日益增加。在各种推荐方法中，基于图的方法因其能有效利用学术网络中固有的结构特征而受到关注。然而，这些方法在学习图表示时往往忽视了学术网络中普遍存在的不对称学术影响。

### 目的

解决传统基于图的推荐方法在学术网络推荐中忽视不对称学术影响的问题，提高科学论文推荐的准确性和质量。

### 方法

提出Mutual-Influence-Aware Recommendation (MIARec)模型，采用基于引力的方法衡量学者之间的相互学术影响，并将这种影响整合到图表示学习中的消息传播过程的特征聚合中。此外，模型利用多通道聚合方法来捕获不同单一关系子网络的个体嵌入及其相互依赖的嵌入，从而能够更全面地理解异构学术网络。

### 主要发现

在真实数据集上进行的大量实验表明，MIARec模型在三个主要评估指标上均优于基线模型，证明了其在科学论文推荐任务中的有效性。

### 结论

MIARec模型通过考虑学者之间的相互学术影响和使用多通道聚合方法，能够更有效地进行科学论文推荐，其性能优于现有的基线模型。

### 翻译

随着科学文献的快速扩张，学者们越来越需要精确和高质量的论文推荐。在各种推荐方法中，基于图的方法通过有效利用学术网络中固有的结构特征而受到关注。然而，这些方法在学习图表示时往往忽视了学术网络中普遍存在的不对称学术影响。为了解决这一局限，本研究提出了相互影响感知推荐(MIARec)模型，该模型采用基于引力的方法来衡量学者之间的相互学术影响，并将这种影响整合到图表示学习过程中消息传播的特征聚合中。此外，该模型利用多通道聚合方法来捕获不同单一关系子网络的个体嵌入及其相互依赖的嵌入，从而能够更全面地理解异构学术网络。在真实数据集上进行的大量实验表明，MIARec模型在三个主要评估指标上均优于基线模型，表明其在科学论文推荐任务中的有效性。


### 论文摘要

With the rapid expansion of scientific literature, scholars increasingly demand precise and high-quality paper recommendations. Among various recommendation methodologies, graph-based approaches have garnered attention by effectively exploiting the structural characteristics inherent in scholarly networks. However, these methods often overlook the asymmetric academic influence that is prevalent in scholarly networks when learning graph representations. To address this limitation, this study proposes the Mutual-Influence-Aware Recommendation (MIARec) model, which employs a gravity-based approach to measure the mutual academic influence between scholars and incorporates this influence into the feature aggregation process during message propagation in graph representation learning. Additionally, the model utilizes a multi-channel aggregation method to capture both individual embeddings of distinct single relational sub-networks and their interdependent embeddings, thereby enabling a more comprehensive understanding of the heterogeneous scholarly network. Extensive experiments conducted on real-world datasets demonstrate that the MIARec model outperforms baseline models across three primary evaluation metrics, indicating its effectiveness in scientific paper recommendation tasks.

---

## 46. TopoAlign: A Framework for Aligning Code to Math via Topological Decomposition

**论文链接:** [http://arxiv.org/abs/2510.11944v1](http://arxiv.org/abs/2510.11944v1)

**作者:** Yupei Li, Philipp Borchert, Gerasimos Lampouras

**发布时间:** 2025-10-13

### GPT解析

### 总结

该研究提出了TopoAlign框架，利用代码仓库作为训练资源来提高大型语言模型在数学自动形式化任务上的性能。

### 背景

大型语言模型在非正式和正式数学推理方面表现出色，但在将非正式数学陈述转换为正式陈述的自动形式化任务上仍有困难。当前数学LLMs的性能受限于包含非正式和正式陈述配对的大规模语料库的稀缺性。

### 目的

解决当前数学LLMs在自动形式化任务上的局限性，利用更广泛可用的代码仓库作为训练资源来提高模型性能。

### 方法

提出TopoAlign框架，将代码分解为文档字符串、主函数和依赖函数，然后重新组装成结构上模仿正式陈述的类似物，产生结构对齐的代码数据用于训练。训练了DeepSeek-Math和Herald两个模型，并在minif2f、Putnam和ProofNet基准上评估。

### 主要发现

TopoAlign显著提升了DeepSeek-Math的性能，在BEq@10上提高17.77%，在typecheck@10上提高68.82%。即使对于专业模型Herald，也在BEq@10和typecheck@10上分别提高了0.12%和1.09%，表明训练结构对齐的代码数据对各种模型都有益。

### 结论

TopoAlign框架成功利用广泛可用的代码仓库作为训练资源，结构对齐的代码数据能有效提高数学LLMs在自动形式化任务上的性能，即使对专业模型也有改进作用。

### 翻译

大型语言模型(LLMs)在非正式和正式(如Lean 4)数学推理方面表现出色，但它们在自动形式化(即将非正式数学陈述转换为正式陈述的任务)方面仍有困难。自动形式化有助于将LLMs的非正式推理与形式证明助手配对，实现机器可验证的生成并减少幻觉。然而，当前数学LLMs的性能受限于大规模语料库的稀缺性，特别是包含非正式和正式陈述配对的语料库。尽管当前模型被训练为从自然语言指令生成代码，但这些代码与形式数学之间的结构和语法差异限制了有效的迁移学习。我们提出了TopoAlign框架，它解锁了广泛可用的代码仓库作为数学LLMs的训练资源。TopoAlign将代码分解为文档字符串、主函数和依赖函数，并将这些组件重新组装成结构上模仿正式陈述的类似物。这产生了结构对齐的代码数据，可用于训练数学LLMs而无需额外的人工注释。我们训练了两个最先进的模型DeepSeek-Math和Herald，并在minif2f、Putnam和ProofNet基准上评估它们。TopoAlign为DeepSeek-Math提供了显著提升，在BEq@10上提高17.77%，在typecheck@10上提高68.82%。尽管没有引入新的数学知识，我们的框架使Herald在BEq@10和typecheck@10上分别提高了0.12%和1.09%，证明了即使在专业模型上，训练结构对齐的代码数据也是有益的。


### 论文摘要

Large Language Models (LLMs) excel at both informal and formal (e.g. Lean 4) mathematical reasoning but still struggle with autoformalisation, the task of transforming informal into formal mathematical statements. Autoformalisation helps pair the informal reasoning of LLMs with formal proof assistants which enable machine-verifiable generation and mitigate hallucinations. Yet, the performance of current Math LLMs is constrained by the scarcity of large-scale corpora, particularly those containing pairs of informal and formal statements. Although current models are trained to generate code from natural language instructions, structural and syntactic differences between these and formal mathematics limit effective transfer learning. We propose TopoAlign, a framework that unlocks widely available code repositories as training resources for Math LLMs. TopoAlign decomposes code into docstrings, main functions, and dependency functions, and reassembles these components into analogues that structurally mirror formal statements. This produces structurally aligned code data that can be used for training Math LLMs without requiring additional human annotation. We train two state-of-the-art models, DeepSeek-Math and Herald, and evaluate them on the minif2f, Putnam, and ProofNet benchmarks. TopoAlign provides substantial gains for DeepSeek-Math, improving performance by 17.77% on BEq@10 and 68.82% on typecheck@10. Despite introducing no new mathematical knowledge, our framework achieves gains of 0.12% and 1.09% for Herald on BEq@10 and typecheck@10, respectively, demonstrating that training on aligned code data is beneficial even for specialized models.

---

## 47. Indoor Localization using Compact, Telemetry-Agnostic, Transfer-Learning Enabled Decoder-Only Transformer

**论文链接:** [http://arxiv.org/abs/2510.11926v1](http://arxiv.org/abs/2510.11926v1)

**作者:** Nayan Sanjay Bhatia, Pranay Kocheta, Russell Elliott, Harikrishna S. Kuttivelil, Katia Obraczka

**发布时间:** 2025-10-13

**备注:** 11 pages, 12 Figures

### GPT解析

### 总结

Locaris是一种仅解码器的大型语言模型，用于室内Wi-Fi定位，能够直接处理原始信号数据而无需预处理，在各种条件下表现出色且无需大量校准。

### 背景

室内Wi-Fi定位具有挑战性，因为无线电信号对环境动态、信道传播特性和硬件异构性高度敏感。传统方法需要密集校准且在条件变化时性能迅速下降。

### 目的

引入Locaris，一种仅解码器的大型语言模型，用于室内定位，解决传统方法的局限性。

### 方法

Locaris将每个接入点测量视为token，摄取原始Wi-Fi遥测数据无需预处理。通过在不同Wi-Fi数据集上微调LLM，学习从原始信号到设备位置的轻量级且可泛化的映射。

### 主要发现

Locaris匹配或超越现有技术；紧凑LLM可作为无校准回归模型；少样本适应实验显示高精度；亚米级精度仅需几百个样本；在缺少AP情况下仍稳健；支持所有可用遥测数据。

### 结论

Locaris在室内定位实际应用中具有实用可行性，特别适用于大规模部署中广泛校准不可行的情况。

### 翻译

室内Wi-Fi定位由于无线电信号对环境动态、信道传播特性和硬件异构性的高度敏感性而仍然是一个具有挑战性的问题。传统的指纹识别和基于模型的方法通常需要密集的校准，并且在设备、信道或部署条件变化时性能迅速下降。在本文中，我们引入了Locaris，一种用于室内定位的仅解码器大型语言模型（LLM）。Locaris将每个接入点（AP）测量视为一个token，能够摄取原始的Wi-Fi遥测数据而无需预处理。通过在不同的Wi-Fi数据集上微调其LLM，Locaris学习从原始信号直接到设备位置的轻量级且可泛化的映射。我们将Locaris与最先进的方法进行比较实验研究，一致表明Locaris在各种类型的遥测数据上匹配或超越现有技术。我们的结果表明，紧凑的LLM可以作为室内定位的无校准回归模型，在异构Wi-Fi部署中提供可扩展和稳健的跨环境性能。使用每个设备仅少数几个校准点的少样本适应实验进一步表明，当应用于未见过的设备和部署场景时，Locaris保持高精度。这仅需几百个样本就能实现亚米级精度，在缺少AP的情况下保持稳健性能，并支持所有可用的遥测数据。我们的发现突显了Locaris在现实场景室内定位中的实际可行性，特别是在广泛校准不可行的大规模部署中。


### 论文摘要

Indoor Wi-Fi positioning remains a challenging problem due to the high sensitivity of radio signals to environmental dynamics, channel propagation characteristics, and hardware heterogeneity. Conventional fingerprinting and model-based approaches typically require labor-intensive calibration and suffer rapid performance degradation when devices, channel or deployment conditions change. In this paper, we introduce Locaris, a decoder-only large language model (LLM) for indoor localization. Locaris treats each access point (AP) measurement as a token, enabling the ingestion of raw Wi-Fi telemetry without pre-processing. By fine-tuning its LLM on different Wi-Fi datasets, Locaris learns a lightweight and generalizable mapping from raw signals directly to device location. Our experimental study comparing Locaris with state-of-the-art methods consistently shows that Locaris matches or surpasses existing techniques for various types of telemetry. Our results demonstrate that compact LLMs can serve as calibration-free regression models for indoor localization, offering scalable and robust cross-environment performance in heterogeneous Wi-Fi deployments. Few-shot adaptation experiments, using only a handful of calibration points per device, further show that Locaris maintains high accuracy when applied to previously unseen devices and deployment scenarios. This yields sub-meter accuracy with just a few hundred samples, robust performance under missing APs and supports any and all available telemetry. Our findings highlight the practical viability of Locaris for indoor positioning in the real-world scenarios, particularly in large-scale deployments where extensive calibration is infeasible.

---

## 48. Schrödinger bridge for generative AI: Soft-constrained formulation and convergence analysis

**论文链接:** [http://arxiv.org/abs/2510.11829v1](http://arxiv.org/abs/2510.11829v1)

**作者:** Jin Ma, Ying Tan, Renyuan Xu

**发布时间:** 2025-10-13

**备注:** 31 pages

### GPT解析

### 总结

本文研究了生成式AI与Schrödinger bridge问题的联系，提出了一种软约束方法来解决经典SBP的稳定性问题。

### 背景

生成式AI可视为学习将简单参考映射为复杂数据分布的模型，与Schrödinger bridge问题有强联系，因两者都通过熵正则化随机动力学在指定边际间插值。

### 目的

解决经典SBP强制执行硬终端约束导致的不稳定性问题，特别是在高维或数据稀缺情况下。

### 方法

采用软约束Schrödinger bridge问题(SCSBP)框架，将终端约束替换为一般惩罚函数，建立McKean-Vlasov类型随机控制公式，并证明随惩罚增加，控制和值函数以线性速率收敛到经典SBP。

### 主要发现

建立了所有惩罚水平下最优解的存在性；首次为软约束桥提供定量收敛保证；揭示惩罚正则化如何实现鲁棒的生成建模、微调和迁移学习。

### 结论

软约束SBP为解决经典SBP在高维和数据稀缺情况下的不稳定性提供了有效方法，对生成式AI领域有重要应用价值。

### 翻译

生成式AI可以被构建为学习将简单参考测度映射为复杂数据分布的问题，最近由于它们通过熵正则化随机动力学在指定边际之间插值的共同特性，它与经典的Schrödinger bridge问题理论有很强的联系。然而，经典SBP强制执行硬终端约束，这往往导致实际实现中的不稳定性，特别是在高维或数据稀缺的情况下。为应对这一挑战，我们遵循所谓的软约束Schrödinger bridge问题的思路，其中终端约束被一般惩罚函数所取代。这种松弛导致更灵活的McKean-Vlasov类型的随机控制公式。我们建立了所有惩罚水平下最优解的存在性，并证明随着惩罚的增加，控制和值函数以线性速率收敛到经典SBP。我们的分析基于Doob的h变换表示、Schrödinger势的稳定性结果、Gamma-收敛以及一种新的固定点论据，该论据将测度空间上的优化问题与辅助的熵最优传输问题耦合。这些结果不仅首次为软约束桥提供了定量收敛保证，还揭示了惩罚正则化如何实现鲁棒的生成建模、微调和迁移学习。


### 论文摘要

Generative AI can be framed as the problem of learning a model that maps simple reference measures into complex data distributions, and it has recently found a strong connection to the classical theory of the Schr\"odinger bridge problems (SBPs) due partly to their common nature of interpolating between prescribed marginals via entropy-regularized stochastic dynamics. However, the classical SBP enforces hard terminal constraints, which often leads to instability in practical implementations, especially in high-dimensional or data-scarce regimes. To address this challenge, we follow the idea of the so-called soft-constrained Schr\"odinger bridge problem (SCSBP), in which the terminal constraint is replaced by a general penalty function. This relaxation leads to a more flexible stochastic control formulation of McKean-Vlasov type.   We establish the existence of optimal solutions for all penalty levels and prove that, as the penalty grows, both the controls and value functions converge to those of the classical SBP at a linear rate. Our analysis builds on Doob's h-transform representations, the stability results of Schr\"odinger potentials, Gamma-convergence, and a novel fixed-point argument that couples an optimization problem over the space of measures with an auxiliary entropic optimal transport problem. These results not only provide the first quantitative convergence guarantees for soft-constrained bridges but also shed light on how penalty regularization enables robust generative modeling, fine-tuning, and transfer learning.

---

## 49. Denoised Diffusion for Object-Focused Image Augmentation

**论文链接:** [http://arxiv.org/abs/2510.08955v2](http://arxiv.org/abs/2510.08955v2)

**作者:** Nisha Pillai, Aditi Virupakshaiah, Harrison W. Smith, Amanda J. Ashworth, Prasanna Gowda, Phillip R. Owens, Adam R. Rivers, Bindu Nanduri, Mahalingam Ramkumar

**发布时间:** 2025-10-10

### GPT解析

### 总结

该研究提出了一种针对数据稀缺条件下动物健康监测的数据增强框架，通过分割动物图像并进行变换和扩散合成，生成多样化场景，提高动物检测和监测性能。

### 背景

现代农业依赖集成监测系统，其中基于无人机的动物健康监测是关键，但面临数据有限、动物体积小、被遮挡或部分可见等问题。迁移学习方法因缺乏反映特定农场条件的大型数据集而效果有限。

### 目的

开发一种针对特定问题、以动物为中心的数据增强策略，专为数据受限条件下的动物健康监测设计，解决数据稀缺与实际应用之间的差距。

### 方法

提出面向对象的数据增强框架，将动物从背景中分割出来，通过变换和基于扩散的合成技术增强图像，创建真实、多样化的场景，以提高动物检测和监测性能。

### 主要发现

初步实验表明，与基线模型相比，增强数据集在动物检测任务上表现更优。通过生成领域特定数据，该方法支持数据稀缺场景下的实时动物健康监测。

### 结论

该数据增强方法能够弥合有限数据与实际应用之间的差距，即使在数据稀缺的情况下也能支持实时动物健康监测解决方案。

### 翻译

现代农业生产操作越来越依赖于集成监测系统，这些系统结合多种数据源以优化农场运营。基于空中无人机的动物健康监测是关键组成部分，但面临数据可用性有限的问题，加上场景特定问题如动物体积小、被遮挡或部分可见。由于缺乏反映特定农场条件（包括动物品种、环境和行为的差异）的大型数据集，迁移学习方法通常无法解决这一限制。因此，需要开发一种针对特定问题、以动物为中心的数据增强策略，专门为这些独特挑战量身定制。为解决这一差距，我们提出了一种面向对象的数据增强框架，专门为数据受限条件下的动物健康监测设计。我们的方法将动物从背景中分割出来，并通过变换和基于扩散的合成来增强它们，创建真实、多样化的场景，以提高动物检测和监测性能。我们的初步实验表明，与基线模型相比，我们的增强数据集在动物检测任务上取得了更好的性能。通过生成领域特定的数据，我们的方法即使在数据稀缺的情况下也能支持实时动物健康监测解决方案，弥合了有限数据与实际应用之间的差距。


### 论文摘要

Modern agricultural operations increasingly rely on integrated monitoring systems that combine multiple data sources for farm optimization. Aerial drone-based animal health monitoring serves as a key component but faces limited data availability, compounded by scene-specific issues such as small, occluded, or partially visible animals. Transfer learning approaches often fail to address this limitation due to the unavailability of large datasets that reflect specific farm conditions, including variations in animal breeds, environments, and behaviors. Therefore, there is a need for developing a problem-specific, animal-focused data augmentation strategy tailored to these unique challenges. To address this gap, we propose an object-focused data augmentation framework designed explicitly for animal health monitoring in constrained data settings. Our approach segments animals from backgrounds and augments them through transformations and diffusion-based synthesis to create realistic, diverse scenes that enhance animal detection and monitoring performance. Our initial experiments demonstrate that our augmented dataset yields superior performance compared to our baseline models on the animal detection task. By generating domain-specific data, our method empowers real-time animal health monitoring solutions even in data-scarce scenarios, bridging the gap between limited data and practical applicability.

---

## 50. TopROI: A topology-informed network approach for tissue partitioning

**论文链接:** [http://arxiv.org/abs/2510.12772v1](http://arxiv.org/abs/2510.12772v1)

**作者:** Sergio Serrano de Haro Iváñez, Joshua W. Moore, Lucile Grzesiak, Eoghan J. Mullholand, Heather Harrington, Simon J. Leedham, Helen M. Byrne

**发布时间:** 2025-10-14

**备注:** 28 pages, 11 Figures

### GPT解析

### 总结

本研究介绍了一种名为TopROI的新方法，用于将点云数据分割为感兴趣区域(ROI)。该方法结合了几何感知网络和持续同调理论，能够同时保留局部几何结构和高级组织架构。在模拟腺体结构和结直肠癌活检数据上验证，该方法优于传统分割方法，能够更好地保留生物学上有意义的结构，并揭示了从健康到癌变的连续性组织变化。

### 背景

哺乳动物组织架构对生物功能至关重要，其破坏是疾病的标志。医学成像技术可生成大型点云数据集捕捉疾病进展中的细胞变化，但传统感兴趣区域(ROI)定义方法基于象限(quadrat-based)，忽略了组织内在结构，可能导致有意义的特征碎片化。

### 目的

开发一种能够保留局部几何结构和高级架构的点云分割方法，用于定义生物学上有意义的ROI，从而更准确地量化组织结构并获取与疾病进展相关结构变化的新见解。

### 方法

TopROI是一种基于拓扑感知和网络的方法，将几何感知网络与持续同调(persistent homology)相结合，利用细胞邻域和多重尺度循环来指导社区检测，从而识别有意义的ROI。

### 主要发现

1) 在模拟腺体结构的合成点云上，TopROI优于传统方法，能维持生物学合理的ROI几何并保留真实结构；2) 在结直肠癌活检数据上，TopROI生成保留类似隐窝结构的ROI，允许进行持续同调分析；3) 研究揭示了从健康黏膜到癌变的连续性组织变化，反映了结构渐进性无序化。

### 结论

TopROI为定义大型点云中生物学上有意义的ROI提供了原则性和灵活的框架，能更准确量化组织结构，并提供与疾病进展相关结构变化的新见解。

### 翻译

哺乳动物组织架构对生物功能至关重要，其破坏是疾病的标志。医学成像技术可以生成大型点云数据集，捕捉疾病进展过程中组织细胞成分的变化。然而，感兴趣区域(ROI)通常基于象限方法定义，这些方法忽略了内在结构，可能导致有意义的特征碎片化。在此，我们介绍TopROI，一种基于拓扑感知的网络方法，用于将点云分割为ROI，同时保留局部几何结构和高级架构。TopROI将几何感知网络与持续同调相结合，利用细胞邻域和多重尺度循环来指导社区检测。应用于模拟腺体结构的合成点云时，TopROI通过维持生物学上合理的ROI几何形状和更好地保留真实结构，优于基于象限和纯粹几何的分割方法。应用于从人类结直肠癌活检获得的细胞点云时，TopROI生成保留类似隐窝结构的ROI，并允许对单个区域进行持续同调分析。本研究揭示了从健康黏膜到癌变的连续性 architectural 变化，反映了组织结构的渐进性无序化。因此，TopROI为定义大型点云中生物学上有意义的ROI提供了一个原则性和灵活的框架，能够更准确地量化组织结构并提供与疾病进展相关的结构变化的新见解。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决医学成像生成的大规模点云数据中如何定义'感兴趣区域'(ROIs)的问题。当前常用的基于网格的分区方法忽略了组织结构的内在特性，可能导致有意义的结构特征被分割。这个问题很重要，因为哺乳动物组织结构对生物功能至关重要，其破坏是疾病的标志，而准确的ROI定义对于研究疾病进展过程中组织结构变化、进行下游分析和诊断至关重要。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到传统网格方法无法保留组织拓扑结构，因此考虑结合计算几何和拓扑数据分析的最新进展。他们借鉴了Delaunay三角剖分来表示局部细胞关系、持久同调技术提取拓扑特征、以及Leiden算法进行社区检测。作者创新性地将这些现有方法整合到一个统一框架中，同时考虑几何和拓扑信息，从而更有效地定义ROI。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合几何信息和拓扑信息来定义ROI，使分区既能保留局部几何结构，又能捕捉更高阶的组织架构。流程包括：1)构建Delaunay几何网络并赋予权重；2)通过持久同调计算拓扑特征并构建拓扑网络；3)整合几何和拓扑网络为一个加权网络；4)使用Leiden算法进行社区检测，最终得到的社区即为定义的ROI。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)创新性地整合几何网络与持久同调特征；2)提供多尺度分析能力；3)能保留如腺体结构等生物学相关结构；4)方法具有可扩展性和灵活性。相比之前工作，TopROI不同于传统网格方法(不考虑组织内在特性)、纯几何方法(无法捕捉高阶结构)和纯拓扑方法(缺乏几何细节)，是首个将计算几何、网络科学和拓扑数据分析整合用于ROI定义的方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'TopROI通过整合几何网络和拓扑数据分析，提供了一种能够同时保留局部几何结构和更高阶组织架构的灵活框架，用于定义生物学上有意义的感兴趣区域，从而更准确地量化组织结构并揭示与疾病进展相关的结构变化。'}


### 论文摘要

Mammalian tissue architecture is central to biological function, and its disruption is a hallmark of disease. Medical imaging techniques can generate large point cloud datasets that capture changes in the cellular composition of such tissues with disease progression. However, regions of interest (ROIs) are usually defined by quadrat-based methods that ignore intrinsic structure and risk fragmenting meaningful features. Here, we introduce TopROI, a topology-informed, network-based method for partitioning point clouds into ROIs that preserves both local geometry and higher-order architecture. TopROI integrates geometry-informed networks with persistent homology, combining cell neighbourhoods and multiscale cycles to guide community detection. Applied to synthetic point clouds that mimic glandular structure, TopROI outperforms quadrat-based and purely geometric partitions by maintaining biologically plausible ROI geometry and better preserving ground-truth structures. Applied to cellular point clouds obtained from human colorectal cancer biopsies, TopROI generates ROIs that preserve crypt-like structures and enable persistent homology analysis of individual regions. This study reveals a continuum of architectural changes from healthy mucosa to carcinoma, reflecting progressive disorganisation in tissue structure. TopROI thus provides a principled and flexible framework for defining biologically meaningful ROIs in large point clouds, enabling more accurate quantification of tissue organization and new insights into structural changes associated with disease progression.

---

## 51. Voronoi-Assisted Diffusion for Computing Unsigned Distance Fields from Unoriented Points

**论文链接:** [http://arxiv.org/abs/2510.12524v1](http://arxiv.org/abs/2510.12524v1)

**作者:** Jiayi Kong, Chen Zong, Junkai Deng, Xuhui Chen, Fei Hou, Shiqing Xin, Junhui Hou, Chen Qian, Ying He

**发布时间:** 2025-10-14

### GPT解析

### 总结

本文提出了一种名为Voronoi-Assisted Diffusion (VAD)的轻量级、无网络方法，用于直接从无方向点云计算无符号距离场(UDF)。该方法通过双向法线分配、法线扩散和UDF梯度场积分实现，能够高效稳定地处理各种复杂几何结构。

### 背景

无符号距离场(UDF)可以表示具有任意拓扑结构的3D形状，包括开放和封闭表面、可定向和不可定向几何以及非流形结构。然而，现有的神经方法在数值稳定性、计算成本和可控性方面存在局限性。

### 目的

开发一种轻量级、计算稳定且高效的方法来直接从无方向点云计算UDF，解决现有神经方法的数值不稳定性和高计算成本问题。

### 方法

VAD方法包括三个主要步骤：(1)基于Voronoi的几何标准通过能量函数引导为输入点分配双向法线；(2)将法线扩散形成近似UDF梯度场；(3)积分梯度场恢复最终UDF。

### 主要发现

VAD能够稳健处理封闭和开放表面，以及复杂的非流形和不可定向几何，同时保持计算效率和稳定性。

### 结论

VAD是一种有效的方法，可以克服现有神经方法在计算UDF时的数值不稳定性和高计算成本问题，同时提供更好的可控性。

### 翻译

无符号距离场(UDF)为具有任意拓扑结构的3D形状提供了灵活的表示，包括开放和封闭表面、可定向和不可定向几何以及非流形结构。虽然最近的神经方法在学习UDF方面显示出潜力，但它们常常面临数值不稳定、计算成本高和可控性有限等问题。我们提出了一种轻量级、无网络的方法Voronoi-Assisted Diffusion (VAD)，用于直接从无方向点云计算UDF。我们的方法首先通过能量函数中编码的两个基于Voronoi的几何标准引导，为输入点分配双向法线以实现最佳对齐。然后将对齐的法线扩散形成近似UDF梯度场，随后积分恢复最终UDF。实验证明，VAD能够稳健处理封闭和开放表面，以及复杂的非流形和不可定向几何，同时保持计算效率和稳定性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从无方向点云计算无符号距离场（UDFs）的问题。这个问题在3D重建、计算机图形学和几何处理等领域非常重要，因为UDFs可以表示具有任意拓扑的3D形状，包括开放和封闭表面、可定向和不可定向几何体以及非流形结构，而现有的神经学习方法往往面临数值不稳定、计算成本高和可控性有限的问题。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到热方法（heat method）的启发，该方法用于在定向数据上计算有符号距离场。他们引入了投影距离场概念，并将其与Voronoi图联系起来，将点方向视为优化变量。作者借鉴了现有工作中的多个方面：热方法用于扩散法线、投影距离场概念、Voronoi图在几何处理中的应用、泊松表面重建框架以及基于广义回转数的方法，但将这些元素组合成一种新的方法来解决UDF计算问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用Voronoi辅助扩散（VAD）框架，首先对输入点分配双向法线，然后将对齐的法线扩散以形成近似的UDF梯度场，最后通过积分恢复最终的UDF。整体流程包括：1) 构建Voronoi图并初始化双向法线；2) 通过最小化Voronoi边界上的不连续性来优化双向法线；3) （可选）对于嘈杂输入，优化点位置；4) 将双向法线扩散并融合成一致的向量场；5) 求解泊松方程重建UDF。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) Voronoi辅助框架优化双向法线；2) 投影距离场概念与Voronoi图结合；3) 将点方向视为优化变量；4) 提出轻量级、无网络方法；5) 能够处理复杂几何结构。相比之前的工作，该方法避免了神经网络的数值不稳定和高计算成本问题；不需要明确的内外区分，能够处理非封闭表面；不需要边界方向，能够处理非流形结构；相比其他UDF方法提供了更好的可控性和稳定性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了Voronoi辅助扩散（VAD）方法，一种从无方向点云计算无符号距离场的轻量级、无网络方法，能够高效准确地处理开放表面、非流形结构和不可定向几何体等复杂情况。'}


### 论文摘要

Unsigned Distance Fields (UDFs) provide a flexible representation for 3D shapes with arbitrary topology, including open and closed surfaces, orientable and non-orientable geometries, and non-manifold structures. While recent neural approaches have shown promise in learning UDFs, they often suffer from numerical instability, high computational cost, and limited controllability. We present a lightweight, network-free method, Voronoi-Assisted Diffusion (VAD), for computing UDFs directly from unoriented point clouds. Our approach begins by assigning bi-directional normals to input points, guided by two Voronoi-based geometric criteria encoded in an energy function for optimal alignment. The aligned normals are then diffused to form an approximate UDF gradient field, which is subsequently integrated to recover the final UDF. Experiments demonstrate that VAD robustly handles watertight and open surfaces, as well as complex non-manifold and non-orientable geometries, while remaining computationally efficient and stable.

---

## 52. Scene Coordinate Reconstruction Priors

**论文链接:** [http://arxiv.org/abs/2510.12387v1](http://arxiv.org/abs/2510.12387v1)

**作者:** Wenjing Bian, Axel Barroso-Laguna, Tommaso Cavallari, Victor Adrian Prisacariu, Eric Brachmann

**发布时间:** 2025-10-14

**备注:** ICCV 2025, Project page: https://nianticspatial.github.io/scr-priors/

### GPT解析

### 总结

本研究提出了一种对场景坐标回归(SCR)模型进行概率性重新解释的方法，通过引入高级重建先验来改善3D场景表示。研究团队探索了多种先验方法，并训练了3D点云扩散模型，这些先验有助于学习更好的场景表示，提高点云质量、配准率和相机姿态，并对下游任务产生积极影响。

### 背景

场景坐标回归(SCR)模型已被证明是3D视觉中强大的隐式场景表示方法，能够实现视觉重定位和运动恢复。然而，这些模型是针对单个场景专门训练的，如果训练图像暗示了不足的多视图约束，SCR模型就会退化。

### 目的

通过引入高级重建先验来改善SCR模型的学习过程，提高场景表示质量，解决在多视图约束不足情况下模型退化的问题。

### 方法

研究团队提出了一种对SCR模型进行概率性重新解释的方法，并探索了多种先验技术：1)简单的深度值分布先验；2)学习合理的场景坐标配置先验；3)在大型室内扫描数据集上训练3D点云扩散模型。这些先验在每个训练步骤中将预测的3D场景点推向合理的几何形状。

### 主要发现

在三个室内数据集上，研究团队发现：1)引入的先验有助于学习更好的场景表示；2)产生了更一致的场景点云；3)提高了配准率；4)改善了相机姿态；5)对下游任务如新视图合成和相机重定位有积极影响。

### 结论

通过概率性重新解释SCR模型并引入高级重建先验，可以显著改善场景表示的质量，即使在多视图约束不足的情况下也能有效工作，从而提高各种3D视觉任务的性能。

### 翻译

场景坐标回归(SCR)模型已被证明是3D视觉中强大的隐式场景表示方法，能够实现视觉重定位和运动恢复。SCR模型是针对单个场景专门训练的。如果训练图像暗示了不足的多视图约束，SCR模型就会退化。我们提出了一种对训练SCR模型进行概率性重新解释的方法，使我们能够注入高级重建先验。我们研究了多种这样的先验，从对重建深度值分布的简单先验，到对合理场景坐标配置的学习先验。对于后者，我们在大型室内扫描语料库上训练了一个3D点云扩散模型。我们的先验在每个训练步骤中将预测的3D场景点推向合理的几何形状，以提高它们的可能性。在三个室内数据集上，我们的先验有助于学习更好的场景表示，产生更一致的场景点云，更高的配准率和更好的相机姿态，对新视图合成和相机重定位等下游任务有积极影响。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决场景坐标回归(SCR)模型在多视图约束不足时的退化问题，特别是在纹理贫乏区域、重复结构等场景下出现的点云分散、相机姿态估计不准确等问题。这个问题在现实中很重要，因为它直接影响室内场景重建质量、相机重定位精度和新视图合成效果，进而影响AR/VR等应用的用户体验；在研究中，它代表了提升神经SfM模型鲁棒性的重要挑战，特别是在缺乏足够视觉重叠的场景中。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了SCR模型在缺乏多视图约束时退化的原因，然后通过概率重新解释SCR训练过程，将重建先验作为负对数似然项融入训练目标。他们设计了三种先验：深度分布先验、深度先验(RGB-D)和3D点云扩散先验。该方法借鉴了ACE框架的架构，从DiffusioNeRF获取灵感但改为在3D空间直接正则化，并首次将扩散模型应用于场景级别的3D点云生成，而非仅限于单个对象。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将SCR训练重新表述为最大似然学习，引入重建先验来引导模型学习更合理的场景表示，使预测的3D场景点朝向合理的几何形状。整体流程包括：1)使用ACE框架进行场景坐标回归，包含特征提取器和回归头；2)将训练目标重新表述为最大化场景坐标的概率，添加负对数先验作为正则化项；3)实现三种先验：深度分布先验、深度先验和点云扩散先验；4)在训练过程中联合优化重投影误差和先验项，扩散先验只在训练迭代5k后应用。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)概率性重新解释SCR训练以融入重建先验；2)提出多种重建先验（深度分布先验、深度先验和3D点云扩散先验）；3)首次将扩散模型应用于场景级别的3D点云生成；4)开发有效的RGB-D版本的ACE。相比之前的工作，本文方法通过高级先验正则化SCR训练，而非简单依赖场景特定训练；与特征匹配方法相比，提供更强的几何约束；与ACE框架相比，联合优化重投影误差和先验项而非交替优化；与3D扩散模型相比，应用于整个室内场景而非单个对象。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过概率性重新解释场景坐标回归训练并引入多种重建先验，显著提升了室内场景重建质量和相机姿态估计准确性，同时保持了测试时的高效性。'}


### 论文摘要

Scene coordinate regression (SCR) models have proven to be powerful implicit scene representations for 3D vision, enabling visual relocalization and structure-from-motion. SCR models are trained specifically for one scene. If training images imply insufficient multi-view constraints SCR models degenerate. We present a probabilistic reinterpretation of training SCR models, which allows us to infuse high-level reconstruction priors. We investigate multiple such priors, ranging from simple priors over the distribution of reconstructed depth values to learned priors over plausible scene coordinate configurations. For the latter, we train a 3D point cloud diffusion model on a large corpus of indoor scans. Our priors push predicted 3D scene points towards plausible geometry at each training step to increase their likelihood. On three indoor datasets our priors help learning better scene representations, resulting in more coherent scene point clouds, higher registration rates and better camera poses, with a positive effect on down-stream tasks such as novel view synthesis and camera relocalization.

---

## 53. IL3D: A Large-Scale Indoor Layout Dataset for LLM-Driven 3D Scene Generation

**论文链接:** [http://arxiv.org/abs/2510.12095v1](http://arxiv.org/abs/2510.12095v1)

**作者:** Wenxu Zhou, Kaixuan Nie, Hang Du, Dong Yin, Wei Huang, Siqiang Guo, Xiaobo Zhang, Pengbo Hu

**发布时间:** 2025-10-14

**备注:** 9 pages main paper; 15 pages references and appendix

### GPT解析

### 总结

本研究提出了IL3D，一个专为大型语言模型驱动的3D场景生成设计的大规模数据集，包含27,816个室内布局和29,215个高保真3D对象资源。

### 背景

室内布局设计领域对多样化、高质量训练数据有迫切需求，现有数据集可能无法满足大型语言模型训练的要求。

### 目的

创建一个支持多模态学习的3D场景生成数据集，用于提升大语言模型在视觉语言任务中的表现，并推动3D场景生成和具身智能的研究。

### 方法

构建包含18种常见房间类型的室内布局数据集，添加实例级自然语言注释，建立严格的评估基准，测试监督微调方法在数据集上的效果。

### 主要发现

在IL3D上对大型语言模型进行监督微调显著提高了模型的泛化能力，性能优于在其他数据集上的微调结果；数据集提供多种多模态数据导出格式，可适应各种视觉任务需求。

### 结论

IL3D作为一个多功能且强大的资源，通过提供高保真场景数据，显著推动了3D场景生成和具身智能的研究进展，特别是支持了具身智能体的环境感知任务。

### 翻译

在这项研究中，我们提出了IL3D，一个精心设计的大型数据集，用于大型语言模型驱动的3D场景生成，解决了室内布局设计中多样化、高质量训练数据的迫切需求。IL3D包含18种常见房间类型的27,816个室内布局和29,215个高保真3D对象资源库，并添加了实例级自然语言注释，以支持视觉语言任务的多模态学习。我们建立了严格的基准来评估LLM驱动的场景生成。实验结果表明，在IL3D上对LLM进行监督微调显著提高了泛化能力，并优于在其他数据集上的SFT性能。IL3D提供灵活的多模态数据导出功能，包括点云、3D边界框、多视图图像、深度图、法线图和语义掩码，能够无缝适应各种视觉任务。作为一个多功能且强大的资源，IL3D通过提供高保真场景数据来支持具身智能体的环境感知任务，显著推动了3D场景生成和具身智能的研究进展。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决缺乏专门为LLM驱动的3D场景生成设计的大规模高质量室内布局数据集问题。这个问题很重要，因为3D室内场景生成是连接具身智能、智能家居设计、虚拟现实交互和机器人环境感知的关键技术，而精确的室内场景建模依赖于高质量的合成数据集，现有数据集在场景多样性、注释完整性和多模态适应性方面存在明显局限。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有数据集的局限性，然后整合了3D-FRONT和HSSD数据集资源，通过人工清理和有针对性的合成数据补充不足。方法设计上采用USD格式使LLM可直接读取场景信息，并使用Qwen3-VL生成详细实例级描述。作者借鉴了现有数据集的经验，整合了3D-FRONT和HSSD的资源，采用HOLODECK方法合成缺失场景类型，并在格式设计和评估指标方面参考了现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个大规模、多样化的室内布局数据集，提供实例级自然语言注释以支持多模态学习，确保室内布局符合现实世界的功能逻辑，并覆盖不同面积和物体密度的室内场景。整体流程包括：整合现有数据集并人工清理；使用HOLODECK方法合成缺失场景类型；将数据转换为USDZ和USDA格式；为对象提供多级注释并使用Qwen3-VL生成详细描述；设计客观和主观评估指标；支持多种数据格式的灵活导出。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：大规模数据集（27,816个室内布局和29,215个对象模型）；覆盖18种常见房间类型；提供实例级自然语言注释；支持多种数据格式导出；使用USD格式实现文本可读性。相比之前工作，IL3D规模更大（超过3D-FRONT和HSSD），提供更全面的注释（大多数现有数据集缺乏自然语言注释），确保更好的功能逻辑，具有更强的多模态适应性，并专为LLM设计使其可直接读取和解析场景信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'IL3D数据集通过提供大规模、多样化的室内布局和丰富的自然语言注释，显著提升了LLM驱动的3D室内场景生成质量和3D感知任务的准确性。'}


### 论文摘要

In this study, we present IL3D, a large-scale dataset meticulously designed for large language model (LLM)-driven 3D scene generation, addressing the pressing demand for diverse, high-quality training data in indoor layout design. Comprising 27,816 indoor layouts across 18 prevalent room types and a library of 29,215 high-fidelity 3D object assets, IL3D is enriched with instance-level natural language annotations to support robust multimodal learning for vision-language tasks. We establish rigorous benchmarks to evaluate LLM-driven scene generation. Experimental results show that supervised fine-tuning (SFT) of LLMs on IL3D significantly improves generalization and surpasses the performance of SFT on other datasets. IL3D offers flexible multimodal data export capabilities, including point clouds, 3D bounding boxes, multiview images, depth maps, normal maps, and semantic masks, enabling seamless adaptation to various visual tasks. As a versatile and robust resource, IL3D significantly advances research in 3D scene generation and embodied intelligence, by providing high-fidelity scene data to support environment perception tasks of embodied agents.

---

## 54. dN/dx Reconstruction with Deep Learning for High-Granularity TPCs

**论文链接:** [http://arxiv.org/abs/2510.10628v2](http://arxiv.org/abs/2510.10628v2)

**作者:** Guang Zhao, Yue Chang, Jinxian Zhang, Linghui Wu, Huirong Qi, Xin She, Mingyi Dong, Shengsen Sun, Jianchun Wang, Yifang Wang, Chunxu Yu

**发布时间:** 2025-10-12

**备注:** 18 pages, 8 figures

### GPT解析

### 总结

本文提出了一种名为Graph Point Transformer (GraphPT)的深度学习模型，用于高粒度时间投影室中的dN/dx重建，以提高粒子识别性能。该模型将TPC数据表示为点云，采用基于图神经网络的U-Net架构，并使用针对点云处理优化的注意力机制。实验表明，GraphPT模型在K/π粒子识别方面比传统方法提高了10%至20%的分离能力。

### 背景

粒子识别(PID)对未来的粒子物理实验如圆形正负电子对撞机和未来圆形对撞机至关重要。高粒度时间投影室(TPC)不仅能提供精确的跟踪，还能实现dN/dx测量用于粒子识别。

### 目的

引入一种深度学习模型Graph Point Transformer (GraphPT)用于dN/dx重建，解决准确重建面临的挑战。

### 方法

将TPC数据表示为点云，采用基于图神经网络的U-Net架构作为网络主干，并融入针对点云处理优化的注意力机制进行节点聚合。

### 主要发现

GraphPT模型在PID性能上超越了传统的截断均值方法，特别是在5到20 GeV/c的动量区间内，K/π分离能力提高了约10%至20%。

### 结论

GraphPT模型是dN/dx重建的有效方法，能显著提高粒子识别性能，对未来粒子物理实验具有重要意义。

### 翻译

粒子识别对于未来的粒子物理实验（如圆形正负电子对撞机和未来圆形对撞机）至关重要。高粒度时间投影室不仅能提供精确的跟踪，还能实现dN/dx测量用于粒子识别。dN/dx方法估计初级电离电子的数量，为PID性能提供了显著改进。然而，准确的重建仍然是该方法面临的主要挑战。在本文中，我们介绍了一种深度学习模型——图点变换器（GraphPT），用于dN/dx重建。在我们的方法中，TPC数据被表示为点云。然后网络主干采用基于图神经网络的U-Net架构，结合了针对点云处理优化的注意力机制进行节点聚合。所提出的GraphPT模型在PID性能上超越了传统的截断均值方法。特别是在5到20 GeV/c的动量区间内，K/π分离能力提高了约10%至20%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决高粒度时间投影室(TPC)中的dN/dx重建挑战。dN/dx方法通过估计初级电离电子数量来提高粒子识别(PID)性能，这对未来粒子物理实验如环形正负电子对撞机至关重要。准确重建面临长漂移距离导致的电子扩散、重叠簇区分困难等问题，解决这些问题能显著提升粒子在高动量区域的区分能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了传统截断均值方法的局限性，然后借鉴了多个现有工作：受点变换器(Point Transformer)启发采用U-Net架构；将TPC数据表示为点云并使用图神经网络(GNN)处理；结合自注意力机制和GNN优势设计GraphPT模型；探索了减法算子和点积算子两种注意力机制，其中点积算子是本文创新。作者通过将轨迹表示为点云，利用图神经网络学习点间关系来区分初级和次级电子。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将TPC电子轨迹表示为点云，利用图神经网络和变换器架构学习点间关系，区分初级和次级电子。流程包括：1)将电子击中点表示为点云；2)构建k最近邻图；3)采用U-Net编码器-解码器结构；4)通过编码器提取高维特征，解码器映射回低维空间；5)融入变换器层使用注意力机制聚合信息；6)输出每个节点概率；7)根据概率和阈值计算dN/dx值。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：提出GraphPT模型；将TPC数据表示为点云并利用GNN处理；设计U-Net架构用于点云；提出点积算子作为新注意力机制；将传统两步重建统一到单一模型；采用端到端训练。不同之处：之前工作主要处理一维波形，本文处理点云数据；之前多用规则方法或简单神经网络，本文使用先进图神经网络和变换器；点积算子比减法算子性能更好；在高粒度TPC上K/π分离能力提升显著。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出基于图点变换器的深度学习方法，通过将TPC数据表示为点云并利用图神经网络和注意力机制，显著提高了高粒度TPC中dN/dx重建的准确性，在K/π粒子识别能力上比传统方法提升了10%到20%。'}


### 论文摘要

Particle identification (PID) is essential for future particle physics experiments such as the Circular Electron-Positron Collider and the Future Circular Collider. A high-granularity Time Projection Chamber (TPC) not only provides precise tracking but also enables dN/dx measurements for PID. The dN/dx method estimates the number of primary ionization electrons, offering significant improvements in PID performance. However, accurate reconstruction remains a major challenge for this approach. In this paper, we introduce a deep learning model, the Graph Point Transformer (GraphPT), for dN/dx reconstruction. In our approach, TPC data are represented as point clouds. The network backbone adopts a U-Net architecture built upon graph neural networks, incorporating an attention mechanism for node aggregation specifically optimized for point cloud processing. The proposed GraphPT model surpasses the traditional truncated mean method in PID performance. In particular, the $K/\pi$ separation power improves by approximately 10% to 20% in the momentum interval from 5 to 20 GeV/c.

---

## 55. Multi-View Graph Learning with Graph-Tuple

**论文链接:** [http://arxiv.org/abs/2510.10341v2](http://arxiv.org/abs/2510.10341v2)

**作者:** Shiyu Chen, Ningyuan Huang, Soledad Villar

**发布时间:** 2025-10-11

**备注:** Submitted to TAG workshop

### GPT解析

### 总结

本文提出了一种多视图图元组框架，用于解决图神经网络在密集图上的效率问题，通过将图划分为不相交的子图来捕捉多尺度交互信息，并在分子性质预测和宇宙学参数推断两个应用中展示了优越性能。

### 背景

图神经网络通常随图边数增加而扩展，适合稀疏图但在密集图(如点云或分子相互作用)上效率较低。常见的稀疏化方法通过相似性阈值或距离修剪强制选择单一交互尺度，丢弃其他尺度的重要信息。

### 目的

克服单一交互尺度的限制，保留多尺度信息，提高图神经网络在密集图上的性能和效率。

### 方法

引入多视图图元组框架，将图划分为不相交的子图捕捉主要局部相互作用和远程连接；通过受非交换算子理论启发的异构消息传递架构学习多视图表示；证明该框架比单图消息传递模型更具表达力并保证更低风险。

### 主要发现

在分子性质预测(从特征稀缺的库仑矩阵)和宇宙学参数推断(从几何点云)两个应用中，多视图图元组模型都表现出比单图基线更好的性能。

### 结论

多视图方法在处理密集图数据时具有强大的功能和通用性，能够有效捕捉多尺度交互信息，提高模型性能。

### 翻译

图神经网络(GNNs)通常随图边数的增加而扩展，使其适合稀疏图但在密集图(如点云或分子相互作用)上效率较低。常见的解决方案是通过相似性阈值或距离修剪来稀疏化图，但这强制选择单一交互尺度并丢弃其他尺度的重要信息。为克服这一限制，我们引入了多视图图元组框架。与单一图不同，我们的图元组框架将图划分为不相交的子图，捕捉主要局部相互作用和较弱的远程连接。然后，我们通过受非交换算子理论启发的异构消息传递架构从图元组中学习多视图表示，我们正式证明这比单图消息传递模型更具表达力，并保证更低的风险。我们在两个科学领域实例化了我们的框架：从特征稀缺的库仑矩阵进行分子性质预测，以及从几何点云进行宇宙学参数推断。在这两种应用中，我们的多视图图元组模型都表现出比单图基线更好的性能，突显了我们多视图方法的强大功能和通用性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决图神经网络（GNNs）在处理密集图（如点云或分子相互作用）时的效率问题。传统方法通过对图进行稀疏化（如相似性阈值化）来提高效率，但这会强制选择单一交互尺度并丢弃其他尺度的重要信息。这个问题在科学和现实应用中很重要，因为许多数据自然表现为密集图结构，如分子中的原子相互作用或宇宙学中的暗物质分布，传统方法会丢失关键信息从而影响模型性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到GNNs在密集图上的计算效率问题和传统稀疏化方法的信息丢失问题。他们考虑了现有解决方案如不变量特征模型和多视图方法，发现这些方法要么依赖低秩假设，要么专为异构图设计。作者设计思路是构建多视图图表示，将单个图根据交互强度划分为强连接图和弱连接图，并受GtNN框架启发，在单层中整合多个消息传递操作。他们借鉴了异构图学习、多视图表示学习和多尺度GNNs的思想，但进行了创新改进以适用于同构图和连续边特征。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是多视图图元组框架，将单个图划分为多个子图（视图）捕捉不同尺度的交互信息，并通过异构消息传递架构同时学习这些视图。实现流程包括：1) 图元组表示：将图G分解为图元组(G1,...,Gk)，每个子图在同一节点集上但具有不相交边集；2) 异构消息传递：在单层中整合尺度内操作（每个图视图内）和尺度间操作（跨不同图视图），公式为H(l+1) = H(l) + Σ(ci·Hi) + Σ(cij·Hi→j + cji·Hj→i)；3) 两种具体实现：GINE-Gt用于一般图，EGNN-Gt用于几何数据。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 多视图图元组框架，同时保留不同尺度的交互信息；2) 异构消息传递架构，建模不同视图间的算子顺序；3) 理论保证，证明框架更具表现力且保证更低风险。相比之前工作，不同之处在于：不丢弃任何尺度信息（vs 传统稀疏化）；不依赖低秩假设（vs 不变量特征模型）；将异构图学习扩展到同构图，基于物理交互强度构建多视图（vs 多视图方法）；在同一节点集上定义多个图，避免跨级别对齐（vs 多尺度GNNs）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出的多视图图元框架通过异构消息传递架构同时学习不同尺度的图交互，解决了图神经网络在密集图上的效率和表示能力之间的权衡问题，并在科学应用中展示了优越性能。'}


### 论文摘要

Graph Neural Networks (GNNs) typically scale with the number of graph edges, making them well suited for sparse graphs but less efficient on dense graphs, such as point clouds or molecular interactions. A common remedy is to sparsify the graph via similarity thresholding or distance pruning, but this forces an arbitrary choice of a single interaction scale and discards crucial information from other scales. To overcome this limitation, we introduce a multi-view graph-tuple framework. Instead of a single graph, our graph-tuple framework partitions the graph into disjoint subgraphs, capturing primary local interactions and weaker, long-range connections. We then learn multi-view representations from the graph-tuple via a heterogeneous message-passing architecture inspired by the theory of non-commuting operators, which we formally prove is strictly more expressive and guarantees a lower oracle risk compared to single-graph message-passing models. We instantiate our framework on two scientific domains: molecular property prediction from feature-scarce Coulomb matrices and cosmological parameter inference from geometric point clouds. On both applications, our multi-view graph-tuple models demonstrate better performance than single-graph baselines, highlighting the power and versatility of our multi-view approach.

---

## 56. Disentangling Neurodegeneration with Brain Age Gap Prediction Models: A Graph Signal Processing Perspective

**论文链接:** [http://arxiv.org/abs/2510.12763v1](http://arxiv.org/abs/2510.12763v1)

**作者:** Saurabh Sihag, Gonzalo Mateos, Alejandro Ribeiro

**发布时间:** 2025-10-14

**备注:** Accepted for publication in IEEE Signal Processing Magazine

### GPT解析

### 总结

本文介绍了脑年龄差距作为神经退行性疾病生物标志物的应用，提出基于图信号处理和图神经网络的方法，特别是协方差神经网络(VNN)，以改进脑年龄差距预测模型的可靠性和可解释性。

### 背景

神经退行性疾病通常通过结构MRI显示的皮层厚度或脑体积减少来评估，但传统方法无法完全捕捉神经退行性病变在空间上的相关性和异质性。脑年龄差距作为一种数据驱动生物标志物虽有潜力，但其实际应用受限于方法学不明确和泛化能力有限。

### 目的

提供BAGP的概述，并基于图信号处理的最新进展引入一个有原则的应用框架，特别是开发协方差神经网络(VNN)以实现稳健的脑年龄差距预测。

### 方法

采用图信号处理、机器学习和网络神经学的综合视角，特别关注图神经网络(GNN)和协方差神经网络(VNN)，后者利用结构MRI推导的解剖协方差矩阵来提供理论基础和操作可解释性。

### 主要发现

脑年龄差距是脑健康的紧凑生物标志物，对疾病进展和严重程度具有预测效用。基于图神经网络和协方差神经网络的方法能够提供强大的理论支持和操作可解释性，实现可靠的脑年龄差距预测。

### 结论

通过整合多学科视角，阐明了可靠和可解释的BAGP模型的发展路径，并指出了个性化医学的未来研究方向。

### 翻译

神经退行性疾病以神经元结构或功能的进行性丧失为特征，临床上通常通过结构MRI显示的皮层厚度或脑体积减少来评估。虽然这些方法提供了信息，但传统方法缺乏足够的统计复杂性来完全捕捉神经退行性病变在空间上的相关性和异质性。为解决这些限制，脑年龄差距已成为一种有前途的脑健康数据驱动生物标志物。脑年龄差距预测模型估计从神经影像数据预测的脑年龄与实际年龄之间的差异。由此产生的脑年龄差距作为脑健康的紧凑生物标志物，最近的研究表明它对疾病进展和严重程度具有预测效用。然而，BAGP模型在实际应用中受到其方法学不明确和泛化能力有限的阻碍。本教程文章概述了BAGP，并基于图信号处理的最新进展，为这一应用引入了一个有原则的框架。特别是，我们关注图神经网络，并引入了协方差神经网络，它利用结构MRI推导的解剖协方差矩阵。VNN提供了坚实的理论基础和操作可解释性，能够实现可靠的脑年龄差距预测。


### 论文摘要

Neurodegeneration, characterized by the progressive loss of neuronal structure or function, is commonly assessed in clinical practice through reductions in cortical thickness or brain volume, as visualized by structural MRI. While informative, these conventional approaches lack the statistical sophistication required to fully capture the spatially correlated and heterogeneous nature of neurodegeneration, which manifests both in healthy aging and in neurological disorders. To address these limitations, brain age gap has emerged as a promising data-driven biomarker of brain health. The brain age gap prediction (BAGP) models estimate the difference between a person's predicted brain age from neuroimaging data and their chronological age. The resulting brain age gap serves as a compact biomarker of brain health, with recent studies demonstrating its predictive utility for disease progression and severity. However, practical adoption of BAGP models is hindered by their methodological obscurities and limited generalizability across diverse clinical populations. This tutorial article provides an overview of BAGP and introduces a principled framework for this application based on recent advancements in graph signal processing (GSP). In particular, we focus on graph neural networks (GNNs) and introduce the coVariance neural network (VNN), which leverages the anatomical covariance matrices derived from structural MRI. VNNs offer strong theoretical grounding and operational interpretability, enabling robust estimation of brain age gap predictions. By integrating perspectives from GSP, machine learning, and network neuroscience, this work clarifies the path forward for reliable and interpretable BAGP models and outlines future research directions in personalized medicine.

---

## 57. CAMNet: Leveraging Cooperative Awareness Messages for Vehicle Trajectory Prediction

**论文链接:** [http://arxiv.org/abs/2510.12703v1](http://arxiv.org/abs/2510.12703v1)

**作者:** Mattia Grasselli, Angelo Porrello, Carlo Augusto Grazia

**发布时间:** 2025-10-14

**备注:** Accepted at the IEEE Consumer Communications & Networking Conference  (CCNC) 2026 - Las Vegas, NV, USA 9 - 12 January 2026

### GPT解析

### 总结

自动驾驶面临安全挑战，车辆传感器存在视野受限问题，车辆间通信特别是协作感知消息(CAM)可有效解决这一问题，本文提出的CAMNet模型证明了CAM数据在车辆轨迹预测中的有效性

### 背景

自动驾驶任务具有挑战性，现代车辆虽配备LiDAR、摄像头和雷达等昂贵传感器，但存在视野和视线可能被其他车辆遮挡的固有局限性，从而降低态势感知能力

### 目的

研究使用协作感知消息(CAM)数据进行车辆轨迹预测，评估CAM数据是否可以被有效利用

### 方法

设计并训练名为CAMNet的神经网络模型，在广泛使用的运动预测数据集上进行训练，并在使用CAM数据从头创建的第二个数据集上进行评估

### 主要发现

CAM数据确实可以支持车辆轨迹预测，CAMNet模型显示出有希望的结果

### 结论

该方法存在一些局限性，这些局限性为未来研究提供了方向

### 翻译

自动驾驶仍然是一项具有挑战性的任务，主要由于安全问题。现代车辆通常配备昂贵的传感器，如LiDAR、摄像头和雷达，以降低事故风险。然而，这些传感器存在固有局限性：它们的视野和视线可能被其他车辆遮挡，从而降低态势感知能力。在此背景下，车辆间通信起着关键作用，因为它使车辆能够共享信息，即使在传感器被遮挡的情况下也能保持彼此的感知。实现这一点的一种方式是通过使用协作感知消息(CAM)。在本文中，我们研究使用CAM数据进行车辆轨迹预测。具体来说，我们在广泛使用的运动预测数据集上设计和训练了一个神经网络——基于协作感知消息的图神经网络(CAMNet)。然后，我们在使用协作感知消息从头创建的第二个数据集上评估该模型，以评估这种类型的数据是否可以被有效利用。我们的方法显示出有希望的结果，表明CAM确实可以支持车辆轨迹预测。同时，我们讨论了该方法的几种局限性，这些局限性突出了未来研究的机会。


### 论文摘要

Autonomous driving remains a challenging task, particularly due to safety concerns. Modern vehicles are typically equipped with expensive sensors such as LiDAR, cameras, and radars to reduce the risk of accidents. However, these sensors face inherent limitations: their field of view and line of sight can be obstructed by other vehicles, thereby reducing situational awareness. In this context, vehicle-to-vehicle communication plays a crucial role, as it enables cars to share information and remain aware of each other even when sensors are occluded. One way to achieve this is through the use of Cooperative Awareness Messages (CAMs). In this paper, we investigate the use of CAM data for vehicle trajectory prediction. Specifically, we design and train a neural network, Cooperative Awareness Message-based Graph Neural Network (CAMNet), on a widely used motion forecasting dataset. We then evaluate the model on a second dataset that we created from scratch using Cooperative Awareness Messages, in order to assess whether this type of data can be effectively exploited. Our approach demonstrates promising results, showing that CAMs can indeed support vehicle trajectory prediction. At the same time, we discuss several limitations of the approach, which highlight opportunities for future research.

---

## 58. PromoGuardian: Detecting Promotion Abuse Fraud with Multi-Relation Fused Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.12652v1](http://arxiv.org/abs/2510.12652v1)

**作者:** Shaofei Li, Xiao Han, Ziqi Zhang, Minyao Hua, Shuli Gao, Zhenkai Liang, Yao Guo, Xiangqun Chen, Ding Li

**发布时间:** 2025-10-14

**备注:** The final version of this paper is going to appear in IEEE Symposium  on Security and Privacy 2026

### GPT解析

### 总结

本研究针对电子商务平台中的促销滥用欺诈问题，提出了PROMOGUARDIAN模型，一种多关系融合图神经网络，通过整合交易数据的空间和时间信息来检测欺诈。实验表明该模型能有效识别促销滥用欺诈行为。

### 背景

随着电子商务平台的发展，欺诈活动日益增多，对平台的安全性和稳定性构成威胁。促销滥用是近年来增长最快的欺诈类型之一，用户通过利用促销活动从平台获取经济利益。

### 目的

研究电子商务平台美团中的促销滥用欺诈问题，并提出有效的检测方法。

### 方法

提出PROMOGUARDIAN，一种新颖的多关系融合图神经网络，将交易数据的空间和时间信息整合到同质图中，以检测促销滥用欺诈。

### 主要发现

促销滥用欺诈是基于群体的欺诈活动，包含囤积和返现滥用两种类型。与传统欺诈不同，它通常涉及普通客户进行合法交易，且两种欺诈活动常常相互交织。

### 结论

在美团真实数据上的实验表明，该模型达到93.15%的精确度，能检测到2.1至5.0倍更多的欺诈者，在生产环境中可防止1.5至8.8倍更多的经济损失，性能优于现有方法。

### 翻译

随着电子商务平台的发展，欺诈活动日益增多，对平台的安全性和稳定性构成重大威胁。促销滥用是近年来增长最快的欺诈类型之一，其特点是用户利用促销活动从平台获取经济利益。为研究此问题，我们对电子商务平台美团进行了首次促销滥用欺诈研究。我们发现促销滥用欺诈是基于群体的欺诈活动，包含囤积和返现滥用两种类型。与虚假评论等传统欺诈活动不同，促销滥用欺诈通常涉及普通客户进行合法交易，且这两种欺诈活动常常相互交织。为解决此问题，我们提议利用空间和时间角度的额外信息来检测促销滥用欺诈。在本文中，我们介绍了PROMOGUARDIAN，一种新颖的多关系融合图神经网络，将交易数据的空间和时间信息整合到同质图中以检测促销滥用欺诈。我们在美团的现实数据上进行了广泛实验，结果表明我们提出的模型在促销滥用欺诈检测方面优于最先进的方法，达到93.15%的精确度，能检测到2.1至5.0倍更多的欺诈者，并在生产环境中防止1.5至8.8倍更多的经济损失。


### 论文摘要

As e-commerce platforms develop, fraudulent activities are increasingly emerging, posing significant threats to the security and stability of these platforms. Promotion abuse is one of the fastest-growing types of fraud in recent years and is characterized by users exploiting promotional activities to gain financial benefits from the platform. To investigate this issue, we conduct the first study on promotion abuse fraud in e-commerce platforms MEITUAN. We find that promotion abuse fraud is a group-based fraudulent activity with two types of fraudulent activities: Stocking Up and Cashback Abuse. Unlike traditional fraudulent activities such as fake reviews, promotion abuse fraud typically involves ordinary customers conducting legitimate transactions and these two types of fraudulent activities are often intertwined. To address this issue, we propose leveraging additional information from the spatial and temporal perspectives to detect promotion abuse fraud. In this paper, we introduce PROMOGUARDIAN, a novel multi-relation fused graph neural network that integrates the spatial and temporal information of transaction data into a homogeneous graph to detect promotion abuse fraud. We conduct extensive experiments on real-world data from MEITUAN, and the results demonstrate that our proposed model outperforms state-of-the-art methods in promotion abuse fraud detection, achieving 93.15% precision, detecting 2.1 to 5.0 times more fraudsters, and preventing 1.5 to 8.8 times more financial losses in production environments.

---

## 59. Enhanced Pre-training of Graph Neural Networks for Million-Scale Heterogeneous Graphs

**论文链接:** [http://arxiv.org/abs/2510.12401v1](http://arxiv.org/abs/2510.12401v1)

**作者:** Shengyin Sun, Chen Ma, Jiehao Chen

**发布时间:** 2025-10-14

**备注:** 26 pages

### GPT解析

### 总结

本文提出了一种在大规模异构图上预训练图神经网络的有效框架，解决了现有方法仅适用于同构图且未考虑语义不匹配的问题。

### 背景

图神经网络促进了图数据挖掘发展，但训练需要大量昂贵且有时不可用的有标签数据。现有自监督预训练方法主要针对同构图设计，而现实世界多为异构图，且未考虑语义不匹配问题。

### 目的

开发一个在大规模异构图上预训练GNNs的有效框架，解决语义不匹配问题并提高模型的可转移性。

### 方法

设计了结构感知的预训练任务捕获异构图结构特性，以及语义感知的预训练任务解决语义不匹配。通过构建由语义邻居组成的扰动子空间，使模型更关注语义空间中的通用知识，学习具有更好可转移性的知识。

### 主要发现

在真实世界大规模异构图上的大量实验表明，所提出的方法优于最先进的基线方法。

### 结论

该框架能有效在大规模异构图上预训练GNNs，解决了现有方法在同构图和语义不匹配方面的局限性。

### 翻译

近年来，图神经网络促进了图数据挖掘的发展。然而，训练GNNs需要足够的有标签任务特定数据，这些数据昂贵且有时不可用。为减少对有标签数据的依赖，最近研究提出通过自监督方式预训练GNNs，然后在有有限标签数据的下游任务中应用预训练的GNNs。然而，大多数现有方法仅针对同构图设计（现实世界中的图大多是异构图），且未考虑语义不匹配问题（原始数据与包含更多可转移语义信息的理想数据之间的语义差异）。本文提出了一种在大规模异构图上预训练GNNs的有效框架。我们首先设计了一个结构感知的预训练任务，旨在捕获异构图中的结构特性。然后，设计了一个语义感知的预训练任务来解决不匹配问题。具体而言，我们构建了一个由语义邻居组成的扰动子空间，帮助处理语义不匹配。语义邻居使模型更专注于语义空间中的通用知识，进而帮助模型学习具有更好可转移性的知识。最后，在真实世界大规模异构图上进行了大量实验，证明了所提出方法优于最先进的基线方法。代码可在https://github.com/sunshy-1/PHE获取。


### 论文摘要

In recent years, graph neural networks (GNNs) have facilitated the development of graph data mining. However, training GNNs requires sufficient labeled task-specific data, which is expensive and sometimes unavailable. To be less dependent on labeled data, recent studies propose to pre-train GNNs in a self-supervised manner and then apply the pre-trained GNNs to downstream tasks with limited labeled data. However, most existing methods are designed solely for homogeneous graphs (real-world graphs are mostly heterogeneous) and do not consider semantic mismatch (the semantic difference between the original data and the ideal data containing more transferable semantic information). In this paper, we propose an effective framework to pre-train GNNs on the large-scale heterogeneous graph. We first design a structure-aware pre-training task, which aims to capture structural properties in heterogeneous graphs. Then, we design a semantic-aware pre-training task to tackle the mismatch. Specifically, we construct a perturbation subspace composed of semantic neighbors to help deal with the semantic mismatch. Semantic neighbors make the model focus more on the general knowledge in the semantic space, which in turn assists the model in learning knowledge with better transferability. Finally, extensive experiments are conducted on real-world large-scale heterogeneous graphs to demonstrate the superiority of the proposed method over state-of-the-art baselines. Code available at https://github.com/sunshy-1/PHE.

---

## 60. Leveraging Teleconnections with Physics-Informed Graph Attention Networks for Long-Range Extreme Rainfall Forecasting in Thailand

**论文链接:** [http://arxiv.org/abs/2510.12328v1](http://arxiv.org/abs/2510.12328v1)

**作者:** Kiattikun Chobtham, Kanoksri Sarinnapakorn, Kritanai Torsri, Prattana Deeprasertkul, Jirawan Kamma

**发布时间:** 2025-10-14

### GPT解析

### 总结

该研究提出了一种结合物理信息的图神经网络与极值分析技术相结合的方法，用于提高泰国地区的降雨预测准确性，特别是对极端事件的预测。

### 背景

准确的降雨预测，特别是极端事件的预测，在气候学和地球系统中仍然是一个重大挑战。

### 目的

开发一种新的方法，结合物理信息的图神经网络与极值分析技术，以提高泰国各气象站的降雨预测能力。

### 方法

使用图结构表示气象站捕捉时空模式，预处理相关气候指标，提出Attention-LSTM模型，使用基于地形降水物理公式的边特征，并通过空间季节感知GPD方法进行POT映射处理极端值。

### 主要发现

实验表明，该方法在大多数地区（包括易发生极端事件的地区）优于成熟的基线方法，并与最先进方法保持竞争力。

### 结论

与业务预测系统SEAS5相比，该方法改进了极端事件的预测，并为生产支持长期水资源管理的精细分辨率地图提供了实际改进。

### 翻译

准确的降雨预测，特别是对于极端事件，在气候学和地球系统中仍然是一个重大挑战。本文提出了一种新颖的物理信息图神经网络(GNNs)结合极值分析技术，以提高泰国各气象站的降雨预测。该模型利用气象站的图结构表示来捕捉复杂的时空模式，并通过遥相关提供可解释性。我们预处理了可能影响区域降雨的相关气候指标。所提出的图注意力网络与长短期记忆网络(Attention-LSTM)应用了注意力机制，使用基于简单地形降水物理公式推导的初始边特征。嵌入随后由LSTM层处理。为解决极值问题，我们使用新颖的空间季节感知广义帕累托分布(GPD)方法进行阈值超限(POT)映射，克服了传统机器学习模型的局限性。实验表明，我们的方法在大多数地区（包括易发生极端事件的地区）优于成熟的基线方法，并与最先进方法保持强劲竞争力。与业务预测系统SEAS5相比，我们的实际应用改进了极端事件的预测，并为生产支持长期水资源管理的精细分辨率地图提供了实际增强。


### 论文摘要

Accurate rainfall forecasting, particularly for extreme events, remains a significant challenge in climatology and the Earth system. This paper presents novel physics-informed Graph Neural Networks (GNNs) combined with extreme-value analysis techniques to improve gauge-station rainfall predictions across Thailand. The model leverages a graph-structured representation of gauge stations to capture complex spatiotemporal patterns, and it offers explainability through teleconnections. We preprocess relevant climate indices that potentially influence regional rainfall. The proposed Graph Attention Network with Long Short-Term Memory (Attention-LSTM) applies the attention mechanism using initial edge features derived from simple orographic-precipitation physics formulation. The embeddings are subsequently processed by LSTM layers. To address extremes, we perform Peak-Over-Threshold (POT) mapping using the novel Spatial Season-aware Generalized Pareto Distribution (GPD) method, which overcomes limitations of traditional machine-learning models. Experiments demonstrate that our method outperforms well-established baselines across most regions, including areas prone to extremes, and remains strongly competitive with the state of the art. Compared with the operational forecasting system SEAS5, our real-world application improves extreme-event prediction and offers a practical enhancement to produce fine-resolution maps that support decision-making in long-term water management.

---

## 61. Using STAR-IRS to Secure Indoor Communications Through Symbol-Level Random Phase Modulation

**论文链接:** [http://arxiv.org/abs/2510.11925v1](http://arxiv.org/abs/2510.11925v1)

**作者:** Yanan Du, Zeyang Sun, Yilan Zhang, Sai Xu, Beiyuan Liu

**发布时间:** 2025-10-13

### GPT解析

### 总结

本文提出了一种基于同时传输与反射智能反射面(STAR-IRS)的安全室内通信方案，通过动态分割电磁波并控制反射和传输信号来增强安全通信性能。

### 背景

在室内通信环境中，发射方(Alice)需要向室内用户(Bob)发送机密信息，同时存在室外窃听者(Eves)的威胁，传统通信方案难以有效保障通信安全。

### 目的

设计一种能够保护传输免受窃听的通信方案，最大化安全速率，并通过硬件加速降低计算延迟。

### 方法

部署STAR-IRS在墙壁或窗户上，将入射电磁波动态分割为透射和反射两个分量；控制反射信号增强Bob的接收质量，对透射信号进行符号级随机相位调制降低Eves的信号质量；提出基于图神经网络(GNN)的方案解决安全速率最大化问题；设计基于FPGA的GNN加速器减少计算延迟。

### 主要发现

所提出的策略在安全性方面优于传统方案和仅反射方案；GNN方法在解决优化问题时比MRT、ZF和MMSE等基准技术取得更优结果；基于FPGA的加速器实现了低推理延迟。

### 结论

STAR-IRS结合GNN和FPGA加速器可以有效提高室内通信的安全性，为安全通信提供了一种高效解决方案。

### 翻译

本文提出了一种基于同时传输与反射智能反射面(STAR-IRS)的安全室内通信方案。具体而言，发射方(Alice)向室内目标用户(Bob)发送机密信息，同时有几个窃听者(Eves)潜伏在外部。为了保护传输免受窃听，在墙壁或窗户上部署了STAR-IRS。当电磁波撞击到STAR-IRS时，入射电磁波被动态分割为两个分量，实现通过表面的传输和表面的反射。反射信号被控制以增强Bob的接收，而透射信号则用符号级随机相移进行调制，以降低Eves的信号质量。基于这种设置，构建了安全速率最大化问题。为解决这一问题，开发了基于图神经网络(GNN)的方案。此外，还设计了一个基于FPGA的GNN加速器以减少计算延迟。仿真结果表明，所提出的策略在安全性方面优于传统方案和仅反射方案。此外，在解决优化问题时，GNN方法比最大比传输(MRT)、迫零(ZF)和最小均方误差(MMSE)等基准技术取得更好的结果。最后，实验评估确认基于FPGA的加速器实现了低推理延迟。


### 论文摘要

This paper proposes a secure indoor communication scheme based on simultaneous transmitting and reflecting intelligent reflecting surface (STAR-IRS). Specifically, a transmitter (Alice) sends confidential information to its intended user (Bob) indoors, while several eavesdroppers (Eves) lurk outside. To safeguard the transmission from eavesdropping, the STAR-IRS is deployed on walls or windows. Upon impinging on the STAR-IRS, the incoming electromagnetic wave is dynamically partitioned into two components, enabling both transmission through and reflection from the surface. The reflected signal is controlled to enhance reception at Bob, while the transmitted signal is modulated with symbol-level random phase shifts to degrade the signal quality at Eves. Based on such a setting, the secrecy rate maximization problem is formulated. To solve it, a graph neural network (GNN)-based scheme is developed. Furthermore, a field-programmable gate array (FPGA)-based GNN accelerator is designed to reduce computational latency. Simulation results demonstrate that the proposed strategy outperforms both the conventional scheme and the reflection-only scheme in terms of secrecy performance. Moreover, the GNN-based approach achieves superior results compared to benchmark techniques such as maximum ratio transmission (MRT), zero forcing (ZF), and minimum mean square error (MMSE) in solving the optimization problem. Finally, experimental evaluations confirm that the FPGA-based accelerator enables low inference latency.

---

