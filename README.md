# 今日论文推荐 - 2026-01-01

共 96 篇论文

---

## 1. Safe in the Future, Dangerous in the Past: Dissecting Temporal and Linguistic Vulnerabilities in LLMs

**论文链接:** [http://arxiv.org/abs/2512.24556v1](http://arxiv.org/abs/2512.24556v1)

**作者:** Muhammad Abdullahi Said, Muhammad Sammani Sani

**发布时间:** 2025-12-31

### GPT解析

### 总结

本研究发现大型语言模型的安全对齐不能简单从英语迁移到其他语言，豪萨语中的安全性表现甚至优于英语，但模型在时间推理方面存在严重缺陷，安全性高度依赖于上下文而非固定属性。

### 背景

大型语言模型正整合到关键全球基础设施中，但安全对齐可从英语零样本迁移到其他语言的假设仍然是一个危险的盲点。

### 目的

对三种最先进模型（GPT-5.1、Gemini 3 Pro和Claude 4.5 Opus）进行系统性审计，测试其在不同语言和时间框架下的安全性表现。

### 方法

使用HausaSafety数据集（基于西非威胁场景的新型对抗性数据集），采用2×4因子设计进行1440次评估，测试英语与豪萨语以及不同时间框架下的安全表现。

### 主要发现

1) 发现'复杂干扰'机制而非简单的多语言安全差距；2) Claude 4.5 Opus在豪萨语中(45.0%)比英语中(36.7%)更安全；3) 存在'时间不对称性'，过去时态仅15.6%安全，未来时态达57.2%安全；4) 最安全与最脆弱配置间有9.2倍差异；5) 模型依赖表面启发式而非语义理解。

### 结论

当前大型语言模型创建了'安全口袋'，使全球南方用户面临本地化伤害风险。提出'不变对齐'作为必要范式转变，以确保在语言和时间变化下的安全稳定性。

### 翻译

随着大型语言模型融入关键全球基础设施，安全对齐可从英语零样本迁移到其他语言的假设仍然是一个危险的盲点。本研究使用HausaSafety（一个基于西非威胁场景的新型对抗性数据集，如Yahoo-Yahoo欺诈、Dane枪支制造等）对三种最先进模型（GPT-5.1、Gemini 3 Pro和Claude 4.5 Opus）进行了系统性审计。通过1440次评估的2×4因子设计，我们测试了英语与豪萨语之间以及时间框架的非线性相互作用。我们的结果挑战了主流的多语言安全差距叙事。我们发现'复杂干扰'机制，安全性由变量交集决定。虽然Claude 4.5 Opus在豪萨语中(45.0%)比英语中(36.7%)更安全（由于不确定性驱动的拒绝），但模型在时间推理方面遭受灾难性失败。我们报告了显著的'时间不对称性'，过去时态框架绕过了防御（15.6%安全），而未来时态场景触发了过度保守的拒绝（57.2%安全）。最安全与最脆弱配置间的9.2倍差异证明了安全不是固定属性而是依赖于上下文的状态。我们得出结论，当前模型依赖表面启发式而非强大语义理解，创建了使全球南方用户面临本地化伤害的'安全口袋'。我们提出'不变对齐'作为确保语言和时间变化下安全稳定性的必要范式转变。


### 论文摘要

As Large Language Models (LLMs) integrate into critical global infrastructure, the assumption that safety alignment transfers zero-shot from English to other languages remains a dangerous blind spot. This study presents a systematic audit of three state of the art models (GPT-5.1, Gemini 3 Pro, and Claude 4.5 Opus) using HausaSafety, a novel adversarial dataset grounded in West African threat scenarios (e.g., Yahoo-Yahoo fraud, Dane gun manufacturing). Employing a 2 x 4 factorial design across 1,440 evaluations, we tested the non-linear interaction between language (English vs. Hausa) and temporal framing. Our results challenge the prevailing multilingual safety gap narrative. Instead of a simple degradation in low-resource settings, we identified a mechanism of Complex Interference where safety is determined by the intersection of variables. While models exhibited a Reverse Linguistic with Claude 4.5 Opus proving significantly safer in Hausa (45.0%) than in English (36.7%) due to uncertainty-driven refusal they suffered catastrophic failures in temporal reasoning. We report a profound Temporal Asymmetry, where past-tense framing bypassed defenses (15.6% safe) while future-tense scenarios triggered hyper-conservative refusals (57.2% safe). The magnitude of this volatility is illustrated by a 9.2x disparity between the safest and most vulnerable configurations, proving that safety is not a fixed property but a context-dependent state. We conclude that current models rely on superficial heuristics rather than robust semantic understanding, creating Safety Pockets that leave Global South users exposed to localized harms. We propose Invariant Alignment as a necessary paradigm shift to ensure safety stability across linguistic and temporal shifts.

---

## 2. Robust Egocentric Referring Video Object Segmentation via Dual-Modal Causal Intervention

**论文链接:** [http://arxiv.org/abs/2512.24323v1](http://arxiv.org/abs/2512.24323v1)

**作者:** Haijing Liu, Zhiyuan Song, Hefeng Wu, Tao Pu, Keze Wang, Liang Lin

**发布时间:** 2025-12-30

**备注:** NeurIPS 2025

### GPT解析

### 总结

本文提出了一种名为Causal Ego-REferring Segmentation (CERES)的新方法，用于解决第一人称视频中基于语言查询的特定物体分割问题。该方法通过因果推理框架解决了现有方法面临的挑战，并在基准测试中取得了最先进的性能。

### 背景

Egocentric Referring Video Object Segmentation (Ego-RVOS)任务旨在理解第一人称视频中人类主动参与的特定物体，但这一任务具有挑战性，因为第一人称视频存在固有的模糊性，训练数据中存在偏差，导致现有方法学习到数据集中的虚假相关性和基本视觉混淆因素。

### 目的

开发一种能够更稳健地进行Ego-RVOS的方法，解决现有方法面临的挑战，包括学习数据集中的虚假相关性和处理第一人称视角的基本视觉混淆因素。

### 方法

引入Causal Ego-REferring Segmentation (CERES)，一种可插拔的因果框架，它将强大的预训练RVOS主干网络适应到第一人称领域。CERES实现了双模态因果干预：应用后门调整原则来抵消从数据集统计中学到的语言表示偏差，利用前门调整概念来解决视觉混淆问题，通过智能融合语义视觉特征和几何深度信息，创建对第一人称扭曲更具鲁棒性的表示。

### 主要发现

CERES在Ego-RVOS基准测试中实现了最先进的性能，展示了应用因果推理构建更可靠模型的潜力。

### 结论

因果推理在构建更广泛的第一人称视频理解模型方面具有潜力，CERES方法为解决Ego-RVOS任务提供了有效解决方案。

### 翻译

以人为中心的指称视频物体分割旨在分割第一人称视频中人类动作所涉及的具体物体，这些物体由语言查询描述。此任务对于理解以人为中心的人类行为至关重要。然而，由于第一人称视频固有的模糊性和训练数据中存在的偏差，稳健地实现此类分割具有挑战性。因此，现有方法常常难以应对，从数据集中的倾斜物体-动作配对学习到虚假相关性，以及第一人称视角的基本视觉混淆因素，如快速运动和频繁遮挡。为解决这些限制，我们引入了因果以人为中心的指称分割，这是一种可插拔的因果框架，将强大的预训练RVOS主干网络适应到以人为中心的领域。CERES实现了双模态因果干预：应用后门调整原则来抵消从数据集统计中学到的语言表示偏差，并利用前门调整概念来解决视觉混淆问题，通过智能融合语义视觉特征和几何深度信息，创建对以人为中心扭曲更具鲁棒性的表示。大量实验表明，CERES在Ego-RVOS基准测试中实现了最先进的性能，突显了应用因果推理构建更可靠模型以进行更广泛的以人为中心的视频理解的潜力。


### 论文摘要

Egocentric Referring Video Object Segmentation (Ego-RVOS) aims to segment the specific object actively involved in a human action, as described by a language query, within first-person videos. This task is critical for understanding egocentric human behavior. However, achieving such segmentation robustly is challenging due to ambiguities inherent in egocentric videos and biases present in training data. Consequently, existing methods often struggle, learning spurious correlations from skewed object-action pairings in datasets and fundamental visual confounding factors of the egocentric perspective, such as rapid motion and frequent occlusions. To address these limitations, we introduce Causal Ego-REferring Segmentation (CERES), a plug-in causal framework that adapts strong, pre-trained RVOS backbones to the egocentric domain. CERES implements dual-modal causal intervention: applying backdoor adjustment principles to counteract language representation biases learned from dataset statistics, and leveraging front-door adjustment concepts to address visual confounding by intelligently integrating semantic visual features with geometric depth information guided by causal principles, creating representations more robust to egocentric distortions. Extensive experiments demonstrate that CERES achieves state-of-the-art performance on Ego-RVOS benchmarks, highlighting the potential of applying causal reasoning to build more reliable models for broader egocentric video understanding.

---

## 3. Taming Hallucinations: Boosting MLLMs' Video Understanding via Counterfactual Video Generation

**论文链接:** [http://arxiv.org/abs/2512.24271v1](http://arxiv.org/abs/2512.24271v1)

**作者:** Zhe Huang, Hao Wen, Aiming Hao, Bingze Song, Meiqi Wu, Jiahong Wu, Xiangxiang Chu, Sheng Lu, Haoqian Wang

**发布时间:** 2025-12-30

**备注:** 18 pages

### GPT解析

### 总结

多模态大语言模型在视频理解方面取得进展，但存在过度依赖语言先验导致视觉幻觉的问题。研究提出DualityForge框架和DualityVidQA数据集，以及DNA-Train训练方法，有效减少了模型幻觉，提升了性能。

### 背景

多模态大语言模型在视频理解方面取得显著进展，但存在过度依赖语言先验的严重漏洞，可能导致视觉幻觉，特别是在处理反常识视频时。问题源于文本和视频之间的内在数据不平衡，且收集和标注反常识数据成本高昂。

### 目的

解决多模态大语言模型过度依赖语言先验的问题，减少模型在处理反常识视频时的视觉幻觉，开发一种低成本的方法来生成反常识训练数据。

### 方法

提出DualityForge，一个反事实数据合成框架，使用可控的基于扩散的视频编辑技术将真实视频转换为反常识场景；构建DualityVidQA大型视频数据集；提出Duality-Normalized Advantage Training (DNA-Train)，一种两阶段的SFT-RL训练方法，在RL阶段应用成对的ℓ1优势归一化。

### 主要发现

在DualityVidQA-Test上的实验表明，该方法显著减少了模型在反常识视频上的幻觉，相比Qwen2.5-VL-7B基线有24.0%的相对提升；在幻觉和通用基准测试中均取得显著提升，显示出强大的泛化能力。

### 结论

成功解决了多模态大语言模型过度依赖语言先验的问题，提供了一种低成本生成反常识训练数据的方法，开源了数据集和代码为后续研究提供支持。

### 翻译

多模态大语言模型在视频理解方面取得了显著进展。然而，它们存在一个严重漏洞：过度依赖语言先验，这可能导致视觉幻觉，特别是在处理违背常识的反事实视频时。由于文本和视频之间的内在数据不平衡，以及收集和标注反事实数据的巨大成本，这一挑战难以解决。为此，我们引入了DualityForge，一个新颖的反事实数据合成框架，它采用可控的基于扩散的视频编辑技术将真实视频转换为反事实场景。通过将结构化上下文信息嵌入视频编辑和问答生成过程，该框架自动生成高质量的问答对以及原始-编辑视频对，用于对比训练。基于此，我们构建了DualityVidQA，一个旨在减少多模态大语言模型幻觉的大型视频数据集。此外，为了充分利用我们成对数据的对比特性，我们提出了Duality-Normalized Advantage Training (DNA-Train)，一种两阶段的SFT-RL训练方法，其中RL阶段应用成对的ℓ1优势归一化，从而实现更稳定和高效的策略优化。在DualityVidQA-Test上的实验表明，我们的方法显著减少了模型在反常识视频上的幻觉，相比Qwen2.5-VL-7B基线有24.0%的相对提升。此外，我们的方法在幻觉和通用基准测试中均取得显著提升，表明其强大的泛化能力。我们将开源我们的数据集和代码。


### 论文摘要

Multimodal Large Language Models (MLLMs) have made remarkable progress in video understanding. However, they suffer from a critical vulnerability: an over-reliance on language priors, which can lead to visual ungrounded hallucinations, especially when processing counterfactual videos that defy common sense. This limitation, stemming from the intrinsic data imbalance between text and video, is challenging to address due to the substantial cost of collecting and annotating counterfactual data. To address this, we introduce DualityForge, a novel counterfactual data synthesis framework that employs controllable, diffusion-based video editing to transform real-world videos into counterfactual scenarios. By embedding structured contextual information into the video editing and QA generation processes, the framework automatically produces high-quality QA pairs together with original-edited video pairs for contrastive training. Based on this, we build DualityVidQA, a large-scale video dataset designed to reduce MLLM hallucinations. In addition, to fully exploit the contrastive nature of our paired data, we propose Duality-Normalized Advantage Training (DNA-Train), a two-stage SFT-RL training regime where the RL phase applies pair-wise $\ell_1$ advantage normalization, thereby enabling a more stable and efficient policy optimization. Experiments on DualityVidQA-Test demonstrate that our method substantially reduces model hallucinations on counterfactual videos, yielding a relative improvement of 24.0% over the Qwen2.5-VL-7B baseline. Moreover, our approach achieves significant gains across both hallucination and general-purpose benchmarks, indicating strong generalization capability. We will open-source our dataset and code.

---

## 4. Factorized Learning for Temporally Grounded Video-Language Models

**论文链接:** [http://arxiv.org/abs/2512.24097v1](http://arxiv.org/abs/2512.24097v1)

**作者:** Wenzheng Zeng, Difei Gao, Mike Zheng Shou, Hwee Tou Ng

**发布时间:** 2025-12-30

**备注:** ICCV 2025 paper. This arXiv version updates Figure 1 to include the concurrent work Qwen2.5-VL to ensure consistency with Table 1

### GPT解析

### 总结

本文提出了一种名为D²VLM的视频-语言模型框架，通过解耦时间定位和文本响应两个任务的学习，并引入分解偏好优化算法，提高了视频理解中事件级感知的准确性。

### 背景

最近的视频-语言模型在视频理解方面显示出巨大潜力，但在事件级感知的准确时间定位方面仍有困难。视频理解中的两个主要因素（时间定位和文本响应）形成逻辑层次结构，但现有工作通常以耦合方式处理这两个任务，导致次优目标。

### 目的

解决现有视频-语言模型在事件级感知的时间定位准确性问题，通过解耦时间定位和文本响应两个任务的学习，提高视频理解的性能。

### 方法

提出D²VLM框架，采用'先定位，再用证据参考回答'的范式，引入证据标记进行证据定位，强调事件级视觉语义捕获。同时引入分解偏好优化(FPO)算法，将概率时间定位建模明确纳入优化目标。此外，构建了一个合成数据集来解决缺乏适合分解偏好学习且具有明确时间定位的数据集问题。

### 主要发现

通过解耦时间定位和文本响应两个任务的学习，并强调它们之间的固有依赖关系，可以显著提高视频理解中事件级感知的准确性。各种任务上的实验证明了该方法的优势。

### 结论

D²VLM框架和分解偏好优化算法有效解决了现有视频-语言模型在事件级感知时间定位方面的困难，提高了视频理解的性能。源代码已在GitHub上公开。

### 翻译

近期的视频-语言模型在视频理解方面展现出巨大潜力，但在事件级感知的准确时间定位方面仍存在挑战。我们观察到视频理解中的两个主要因素（即时间定位和文本响应）形成了一个逻辑层次结构：准确的时间证据定位是可靠文本响应的基础。然而，现有工作通常以耦合方式处理这两个任务，缺乏明确的逻辑结构，导致次优目标。我们从分解学习的角度解决这个问题。首先提出了D²VLM框架，该框架解耦了这两个任务的学习，同时强调它们固有的依赖关系。我们采用'先定位，再用证据参考回答'的范式，并引入证据标记进行证据定位，强调事件级视觉语义捕获，超越了现有工作中对时间戳表示的关注。为了进一步促进这两个任务的学习，我们引入了一种新颖的分解偏好优化(FPO)算法。与标准偏好优化不同，FPO将概率时间定位建模明确纳入优化目标，实现了时间定位和文本响应的偏好学习。我们还构建了一个合成数据集，以解决缺乏适合分解偏好学习且具有明确时间定位的合适数据集的问题。在各种任务上的实验证明了我们方法的明显优势。我们的源代码可在https://github.com/nusnlp/d2vlm获取。


### 论文摘要

Recent video-language models have shown great potential for video understanding, but still struggle with accurate temporal grounding for event-level perception. We observe that two main factors in video understanding (i.e., temporal grounding and textual response) form a logical hierarchy: accurate temporal evidence grounding lays the foundation for reliable textual response. However, existing works typically handle these two tasks in a coupled manner without a clear logical structure, leading to sub-optimal objectives. We address this from a factorized learning perspective. We first propose D$^2$VLM, a framework that decouples the learning of these two tasks while also emphasizing their inherent dependency. We adopt a "grounding then answering with evidence referencing" paradigm and introduce evidence tokens for evidence grounding, which emphasize event-level visual semantic capture beyond the focus on timestamp representation in existing works. To further facilitate the learning of these two tasks, we introduce a novel factorized preference optimization (FPO) algorithm. Unlike standard preference optimization, FPO explicitly incorporates probabilistic temporal grounding modeling into the optimization objective, enabling preference learning for both temporal grounding and textual response. We also construct a synthetic dataset to address the lack of suitable datasets for factorized preference learning with explicit temporal grounding. Experiments on various tasks demonstrate the clear advantage of our approach. Our source code is available at https://github.com/nusnlp/d2vlm.

---

## 5. AHA: Aligning Large Audio-Language Models for Reasoning Hallucinations via Counterfactual Hard Negatives

**论文链接:** [http://arxiv.org/abs/2512.24052v1](http://arxiv.org/abs/2512.24052v1)

**作者:** Yanxi Chen, Wenhui Zhu, Xiwen Chen, Zhipeng Wang, Xin Li, Peijie Qiu, Hao Wang, Xuanzhao Dong, Yujian Xiong, Anderson Schneider, Yuriy Nevmyvaka, Yalin Wang

**发布时间:** 2025-12-30

### GPT解析

### 总结

该论文提出了AHA（Audio Hallucination Alignment）框架，通过反事实硬负挖掘构建高质量偏好数据集，解决了大型音频语言模型中的幻觉问题，显著提高了模型性能。

### 背景

大型音频语言模型（LALMs）虽然达到最先进性能，但经常出现幻觉问题，即生成不基于音频输入的文本。

### 目的

解决音频语言模型中的幻觉问题，特别是那些与音频输入不相关的内容生成问题。

### 方法

分析幻觉问题并确定分类法（事件省略、错误事件身份、时间关系错误和定量时间错误）；引入AHA框架；利用反事实硬负挖掘构建高质量偏好数据集；建立AHA-Eval诊断基准；将数据应用于对齐Qwen2.5-Omni模型，创建Qwen-Audio-AHA模型。

### 主要发现

Qwen-Audio-AHA模型在AHA-Eval上实现13.7%的改进；这种改进可推广到公共基准测试，在MMAU-Test上提高1.3%，在MMAR上提高1.6%，超过最新最先进方法。

### 结论

AHA框架有效解决了音频语言模型中的幻觉问题，通过构建高质量偏好数据集和对齐模型，显著提高了模型性能，且改进可推广到其他公共基准测试。

### 翻译

虽然大型音频语言模型（LALMs）提供了最先进的性能，但它们经常遭受幻觉问题，例如生成不基于音频输入的文本。我们分析了这些基础失败问题，并确定了一个明确的分类法：事件省略、错误事件身份、时间关系错误和定量时间错误。为解决这一问题，我们引入了AHA（Audio Hallucination Alignment）框架。通过利用反事实硬负挖掘，我们的流程构建了一个高质量偏好数据集，迫使模型区分严格的声学证据和语言上合理的虚构内容。此外，我们建立了AHA-Eval，一个旨在严格测试这些细粒度时间推理能力的诊断基准。我们将这些数据应用于对齐Qwen2.5-Omni。 resulting模型Qwen-Audio-AHA在AHA-Eval上实现了13.7%的改进。重要的是，这种好处超越了我们的诊断集。我们的模型在公共基准测试上显示出显著提升，包括在MMAU-Test上提高1.3%，在MMAR上提高1.6%，超过了最新的最先进方法。


### 论文摘要

Although Large Audio-Language Models (LALMs) deliver state-of-the-art (SOTA) performance, they frequently suffer from hallucinations, e.g. generating text not grounded in the audio input. We analyze these grounding failures and identify a distinct taxonomy: Event Omission, False Event Identity, Temporal Relation Error, and Quantitative Temporal Error. To address this, we introduce the AHA (Audio Hallucination Alignment) framework. By leveraging counterfactual hard negative mining, our pipeline constructs a high-quality preference dataset that forces models to distinguish strict acoustic evidence from linguistically plausible fabrications. Additionally, we establish AHA-Eval, a diagnostic benchmark designed to rigorously test these fine-grained temporal reasoning capabilities. We apply this data to align Qwen2.5-Omni. The resulting model, Qwen-Audio-AHA, achieves a 13.7% improvement on AHA-Eval. Crucially, this benefit generalizes beyond our diagnostic set. Our model shows substantial gains on public benchmarks, including 1.3% on MMAU-Test and 1.6% on MMAR, outperforming latest SOTA methods.

---

## 6. Efficient Deep Learning for Short-Term Solar Irradiance Time Series Forecasting: A Benchmark Study in Ho Chi Minh City

**论文链接:** [http://arxiv.org/abs/2512.23898v1](http://arxiv.org/abs/2512.23898v1)

**作者:** Tin Hoang

**发布时间:** 2025-12-29

**备注:** preprint, 40 pages

### GPT解析

### 总结

本研究对胡志明市短期全球水平辐照量预测进行了十种深度学习架构的全面基准测试，发现Transformer架构表现最佳，并通过知识蒸馏实现了模型压缩，为边缘设备部署高效预测提供了可行方案。

### 背景

全球水平辐照量的可靠预测对于缓解电网中太阳能的变异性至关重要。

### 目的

对胡志明市短期(提前1小时)GHI时间序列预测的十种深度学习架构进行全面基准测试。

### 方法

利用高分辨率NSRDB卫星数据(2011-2020年)，比较基线架构(如LSTM、TCN)与新兴的最先进架构(包括Transformer、Informer、iTransformer、TSMixer和Mamba)，并使用SHAP分析对比这些架构的时间推理能力。

### 主要发现

Transformer被确定为最佳架构，R^2达到0.9696；Transformer表现出强烈的'近期偏见'，专注于当前天气条件；Mamba明确利用24小时的周期性依赖关系；知识蒸馏可将Transformer压缩23.5%，同时减少误差(MAE: 23.78 W/m^2)。

### 结论

知识蒸馏为在资源受限的边缘设备上部署复杂、低延迟的预测提供了可行途径。

### 翻译

全球水平辐照量的可靠预测对于缓解电网中太阳能的变异性至关重要。本研究利用高分辨率NSRDB卫星数据(2011-2020年)，对胡志明市短期(提前1小时)GHI时间序列预测的十种深度学习架构进行了全面基准测试，比较了基线架构(如LSTM、TCN)与新兴的最先进架构(包括Transformer、Informer、iTransformer、TSMixer和Mamba)。实验结果表明Transformer是最佳架构，实现了最高的预测准确性，R^2达到0.9696。研究进一步利用SHAP分析对比了这些架构的时间推理能力，揭示Transformer表现出强烈的'近期偏见'，专注于当前的天气条件，而Mamba明确利用24小时的周期性依赖关系来支持预测。此外，我们证明了知识蒸馏可以将高性能的Transformer压缩23.5%，同时意外地减少了误差(MAE: 23.78 W/m^2)，为在资源受限的边缘设备上部署复杂、低延迟的预测提供了经过验证的途径。


### 论文摘要

Reliable forecasting of Global Horizontal Irradiance (GHI) is essential for mitigating the variability of solar energy in power grids. This study presents a comprehensive benchmark of ten deep learning architectures for short-term (1-hour ahead) GHI time series forecasting in Ho Chi Minh City, leveraging high-resolution NSRDB satellite data (2011-2020) to compare established baselines (e.g. LSTM, TCN) against emerging state-of-the-art architectures, including Transformer, Informer, iTransformer, TSMixer, and Mamba. Experimental results identify the Transformer as the superior architecture, achieving the highest predictive accuracy with an R^2 of 0.9696. The study further utilizes SHAP analysis to contrast the temporal reasoning of these architectures, revealing that Transformers exhibit a strong "recency bias" focused on immediate atmospheric conditions, whereas Mamba explicitly leverages 24-hour periodic dependencies to inform predictions. Furthermore, we demonstrate that Knowledge Distillation can compress the high-performance Transformer by 23.5% while surprisingly reducing error (MAE: 23.78 W/m^2), offering a proven pathway for deploying sophisticated, low-latency forecasting on resource-constrained edge devices.

---

## 7. RoboMirror: Understand Before You Imitate for Video to Humanoid Locomotion

**论文链接:** [http://arxiv.org/abs/2512.23649v2](http://arxiv.org/abs/2512.23649v2)

**作者:** Zhe Li, Cheng Chi, Boan Zhu, Yangyang Wei, Shuanghao Bai, Yuheng Ji, Yibo Peng, Tao Huang, Pengwei Wang, Zhongyuan Wang, S. -H. Gary Chan, Chang Xu, Shanghang Zhang

**发布时间:** 2025-12-29

### GPT解析

### 总结

本文提出RoboMirror，首个无需重定位的视频到运动框架，通过视觉语言模型将视频提炼为视觉运动意图，直接调节基于扩散的策略生成物理合理且语义对齐的运动，有效弥合了视觉理解和控制之间的差距。

### 背景

人类通过视觉观察学习运动，先解释视觉内容再模仿动作。然而，当前最先进的人形运动系统依赖于精心制作动作捕捉轨迹或稀疏文本命令，在视觉理解和控制之间存在关键差距。

### 目的

开发一种能够真正理解视频内容并生成相应人形运动的系统，解决文本到运动方法的语义稀疏问题以及基于视频方法只进行机械姿态模仿而没有真正视觉理解的问题。

### 方法

提出RoboMirror框架，利用视觉语言模型(VLMs)将原始的第一人称/第三人称视频提炼为视觉运动意图，这些意图直接调节基于扩散的策略，生成物理合理、语义对齐的运动，无需显式姿态重建或重定位。

### 主要发现

RoboMirror可以通过第一人称视频实现远程呈现，将第三人称控制延迟减少80%，比基线方法高3.7%的任务成功率，验证了其有效性。

### 结论

通过围绕视频理解重新构建人形控制，RoboMirror成功弥合了视觉理解和行动之间的差距，实现了'先理解再模仿'的理念。

### 翻译

人类通过视觉观察学习运动，先解释视觉内容再模仿动作。然而，最先进的人形运动系统依赖于精心制作动作捕捉轨迹或稀疏文本命令，在视觉理解和控制之间留下关键差距。文本到运动方法受语义稀疏和流水线错误影响，而基于视频的方法只进行机械姿态模仿，没有真正的视觉理解。我们提出RoboMirror，首个无需重定位的视频到运动框架，体现了'先理解再模仿'的理念。利用视觉语言模型，它将原始的第一人称/第三人称视频提炼为视觉运动意图，这些意图直接调节基于扩散的策略，生成物理合理、语义对齐的运动，无需显式姿态重建或重定位。大量实验验证了RoboMirror的有效性，它可以通过第一人称视频实现远程呈现，将第三人称控制延迟减少80%，比基线方法高3.7%的任务成功率。通过围绕视频理解重新构建人形控制，我们弥合了视觉理解和行动之间的差距。


### 论文摘要

Humans learn locomotion through visual observation, interpreting visual content first before imitating actions. However, state-of-the-art humanoid locomotion systems rely on either curated motion capture trajectories or sparse text commands, leaving a critical gap between visual understanding and control. Text-to-motion methods suffer from semantic sparsity and staged pipeline errors, while video-based approaches only perform mechanical pose mimicry without genuine visual understanding. We propose RoboMirror, the first retargeting-free video-to-locomotion framework embodying "understand before you imitate". Leveraging VLMs, it distills raw egocentric/third-person videos into visual motion intents, which directly condition a diffusion-based policy to generate physically plausible, semantically aligned locomotion without explicit pose reconstruction or retargeting. Extensive experiments validate the effectiveness of RoboMirror, it enables telepresence via egocentric videos, drastically reduces third-person control latency by 80%, and achieves a 3.7% higher task success rate than baselines. By reframing humanoid control around video understanding, we bridge the visual understanding and action gap.

---

## 8. TabMixNN: A Unified Deep Learning Framework for Structural Mixed Effects Modeling on Tabular Data

**论文链接:** [http://arxiv.org/abs/2512.23787v1](http://arxiv.org/abs/2512.23787v1)

**作者:** Deniz Akdemir

**发布时间:** 2025-12-29

### GPT解析

### 总结

TabMixNN是一个基于PyTorch的灵活深度学习框架，结合了经典混合效应模型和现代神经网络架构，用于处理表格数据分析，特别是层次数据结构，支持多种结果类型。

### 背景

随着数据分析需求的增长，需要能够处理层次数据结构并支持多种结果类型（回归、分类、多任务学习）的方法。

### 目的

开发一个能够结合混合效应模型和神经网络优势的框架，用于表格数据分析，同时保持可解释性和理论基础。

### 方法

实现了一个模块化的三阶段架构：(1)具有变分随机效应和灵活协方差结构的混合效应编码器，(2)包括广义结构方程模型和时空流形网络的主干架构，(3)支持多种结果族的特定结果预测头。主要创新包括R风格公式接口、DAG约束支持、SPDE核空间建模和全面的可解释性工具。

### 主要发现

通过纵向数据分析、基因预测和时空建模应用，展示了TabMixNN框架的灵活性和有效性。

### 结论

TabMixNN为研究人员提供了一个统一接口，可以利用深度学习的强大功能，同时保持经典混合效应模型的可解释性和理论基础。

### 翻译

我们提出了TabMixNN，这是一个灵活的基于PyTorch的深度学习框架，它将经典的混合效应建模与现代神经网络架构相结合，用于表格数据分析。TabMixNN满足了处理层次数据结构并支持多种结果类型（包括回归、分类和多任务学习）的日益增长的需求。该框架实现了模块化的三阶段架构：(1)具有变分随机效应和灵活协方差结构的混合效应编码器，(2)包括广义结构方程模型(GSEM)和时空流形网络的主干架构，(3)支持多种结果族的特定结果预测头。主要创新包括R风格公式接口（提高可访问性）、支持有向无环图(DAG)约束（用于因果结构学习）、随机偏微分方程(SPDE)核（用于空间建模）以及全面的可解释性工具，包括SHAP值和方差分解。我们通过纵向数据分析、基因预测和时空建模应用展示了该框架的灵活性。TabMixNN为研究人员提供了一个统一接口，可以利用深度学习，同时保持经典混合效应模型的可解释性和理论基础。


### 论文摘要

We present TabMixNN, a flexible PyTorch-based deep learning framework that synthesizes classical mixed-effects modeling with modern neural network architectures for tabular data analysis. TabMixNN addresses the growing need for methods that can handle hierarchical data structures while supporting diverse outcome types including regression, classification, and multitask learning. The framework implements a modular three-stage architecture: (1) a mixed-effects encoder with variational random effects and flexible covariance structures, (2) backbone architectures including Generalized Structural Equation Models (GSEM) and spatial-temporal manifold networks, and (3) outcome-specific prediction heads supporting multiple outcome families. Key innovations include an R-style formula interface for accessibility, support for directed acyclic graph (DAG) constraints for causal structure learning, Stochastic Partial Differential Equation (SPDE) kernels for spatial modeling, and comprehensive interpretability tools including SHAP values and variance decomposition. We demonstrate the framework's flexibility through applications to longitudinal data analysis, genomic prediction, and spatial-temporal modeling. TabMixNN provides a unified interface for researchers to leverage deep learning while maintaining the interpretability and theoretical grounding of classical mixed-effects models.

---

## 9. A Context-Aware Temporal Modeling through Unified Multi-Scale Temporal Encoding and Hierarchical Sequence Learning for Single-Channel EEG Sleep Staging

**论文链接:** [http://arxiv.org/abs/2512.22976v2](http://arxiv.org/abs/2512.22976v2)

**作者:** Amirali Vakili, Salar Jahanshiri, Armin Salimi-Badr

**发布时间:** 2025-12-28

### GPT解析

### 总结

该研究提出了一种用于单通道EEG睡眠分期的上下文感知和可解释框架，特别改进了N1阶段的检测。通过结合多尺度特征提取和时间建模，并应用类别加权损失函数和数据增强来解决数据不平衡问题，该方法在SleepEDF数据集上实现了89.72%的整体准确率，N1阶段的F1分数达到61.7%，显著优于先前方法，同时保持了解释性和临床适用性。

### 背景

自动睡眠分期在医疗保健中至关重要，因为睡眠障碍在全球范围内普遍存在。单通道脑电图(EEG)是一种实用且广泛可用的信号，用于自动睡眠分期。现有方法面临挑战，如类别不平衡、感受野建模有限和可解释性不足。许多先前的模型作为黑盒运行，缺乏明确定义和可解释的特征提取角色。

### 目的

提出一个上下文感知且可解释的框架，用于单通道EEG睡眠分期，特别强调改进N1阶段的检测，解决现有模型缺乏可解释性的问题。

### 方法

结合紧凑的多尺度特征提取与时间建模，以捕捉局部和长程依赖关系。为解决数据不平衡问题，特别是在N1阶段，采用类别加权损失函数和数据增强。将EEG信号分割为子时段块，通过跨块平均softmax概率获得最终预测，增强上下文表示和鲁棒性。

### 主要发现

所提出的框架实现了89.72%的整体准确率和85.46%的宏平均F1分数。对于具有挑战性的N1阶段，达到了61.7%的F1分数，在SleepEDF数据集上相比之前的方法有显著改进。

### 结论

所提出的方法有效提高了睡眠分期性能，同时保持了可解释性，适合实际临床应用。

### 翻译

自动睡眠分期是医疗保健中的一个关键任务，因为睡眠障碍在全球范围内普遍存在。本研究专注于单通道脑电图(EEG)，这是一种实用且广泛可用的信号，用于自动睡眠分期。现有方法面临诸如类别不平衡、有限感受野建模和不足可解释性等挑战。这项工作提出了一个用于单通道EEG睡眠分期的上下文感知和可解释框架，特别强调改进N1阶段的检测。许多先前的模型作为具有堆叠层的黑盒运行，缺乏明确定义和可解释的特征提取角色。所提出的模型结合了紧凑的多尺度特征提取与时间建模，以捕捉局部和长程依赖关系。为解决数据不平衡问题，特别是在N1阶段，应用了类别加权损失函数和数据增强。EEG信号被分割为子时段块，通过跨块平均softmax概率获得最终预测，增强了上下文表示和鲁棒性。所提出的框架实现了89.72%的整体准确率和85.46%的宏平均F1分数。值得注意的是，它在具有挑战性的N1阶段达到了61.7%的F1分数，在SleepEDF数据集上相比先前方法显示出显著改进。这些结果表明，所提出的方法在保持可解释性和适合实际临床应用的同时，有效提高了睡眠分期性能。


### 论文摘要

Automatic sleep staging is a critical task in healthcare due to the global prevalence of sleep disorders. This study focuses on single-channel electroencephalography (EEG), a practical and widely available signal for automatic sleep staging. Existing approaches face challenges such as class imbalance, limited receptive-field modeling, and insufficient interpretability. This work proposes a context-aware and interpretable framework for single-channel EEG sleep staging, with particular emphasis on improving detection of the N1 stage. Many prior models operate as black boxes with stacked layers, lacking clearly defined and interpretable feature extraction roles.The proposed model combines compact multi-scale feature extraction with temporal modeling to capture both local and long-range dependencies. To address data imbalance, especially in the N1 stage, classweighted loss functions and data augmentation are applied. EEG signals are segmented into sub-epoch chunks, and final predictions are obtained by averaging softmax probabilities across chunks, enhancing contextual representation and robustness.The proposed framework achieves an overall accuracy of 89.72% and a macro-average F1-score of 85.46%. Notably, it attains an F1- score of 61.7% for the challenging N1 stage, demonstrating a substantial improvement over previous methods on the SleepEDF datasets. These results indicate that the proposed approach effectively improves sleep staging performance while maintaining interpretability and suitability for real-world clinical applications.

---

## 10. AUDRON: A Deep Learning Framework with Fused Acoustic Signatures for Drone Type Recognition

**论文链接:** [http://arxiv.org/abs/2512.20407v2](http://arxiv.org/abs/2512.20407v2)

**作者:** Rajdeep Chatterjee, Sudip Chakrabarty, Trishaani Acharjee, Deepanjali Mishra

**发布时间:** 2025-12-23

**备注:** Presented at the 2025 IEEE 22nd India Council International Conference (INDICON). 6 pages, 3 figures

### GPT解析

### 总结

本研究提出了AUDRON框架，一种基于音频的无人机识别网络，通过结合多种声学特征和深度学习方法实现高精度的无人机检测。

### 背景

无人机(UAVs)在物流、农业、监控和国防等领域被广泛使用，但其滥用带来了安全和安全问题，因此有效的检测机制至关重要。

### 目的

开发一种低成本的无人机检测方法，专注于声学传感，以替代视觉或雷达检测方案。

### 方法

AUDRON框架结合了梅尔频率倒谱系数(MFCC)、短时傅里叶变换(STFT)频谱图（使用卷积神经网络处理）、用于时间建模的循环层和基于自编码器的表示，并通过特征级融合整合互补信息。

### 主要发现

AUDRON能够有效区分无人机的声学特征与背景噪声，在二进制分类和多类分类中分别达到98.51%和97.11%的准确率，同时保持跨条件的泛化能力。

### 结论

结合多种特征表示和深度学习对于可靠的声音无人机检测具有显著优势，该框架在视觉或雷达传感可能受限的安全和监控应用中具有实际部署潜力。

### 翻译

无人机（UAVs），通常被称为无人机，正被广泛应用于物流、农业、监控和国防等多个领域。尽管这些系统提供了诸多好处，但其滥用引发了安全和安全问题，使得有效的检测机制变得必不可少。声学传感提供了一种低成本和非侵入性的替代方案，用于基于视觉或雷达的检测，因为无人机螺旋桨会产生独特的声音模式。本研究引入了AUDRON（基于音频的无人机识别网络），这是一种用于无人机声音检测的混合深度学习框架，采用梅尔频率倒谱系数（MFCC）、短时傅里叶变换（STFT）频谱图（使用卷积神经网络CNN处理）、用于时间建模的循环层和基于自编码器的表示相结合的方法。特征级融合在分类前整合了互补信息。实验评估表明，AUDRON能够有效区分无人机声学特征与背景噪声，在保持跨条件泛化能力的同时实现高精度。AUDRON在二进制分类和多类分类中分别达到98.51%和97.11%的准确率。结果突显了结合多种特征表示和深度学习进行可靠声学无人机检测的优势，表明该框架在视觉或雷达传感可能受限的安全和监控应用中具有部署潜力。


### 论文摘要

Unmanned aerial vehicles (UAVs), commonly known as drones, are increasingly used across diverse domains, including logistics, agriculture, surveillance, and defense. While these systems provide numerous benefits, their misuse raises safety and security concerns, making effective detection mechanisms essential. Acoustic sensing offers a low-cost and non-intrusive alternative to vision or radar-based detection, as drone propellers generate distinctive sound patterns. This study introduces AUDRON (AUdio-based Drone Recognition Network), a hybrid deep learning framework for drone sound detection, employing a combination of Mel-Frequency Cepstral Coefficients (MFCC), Short-Time Fourier Transform (STFT) spectrograms processed with convolutional neural networks (CNNs), recurrent layers for temporal modeling, and autoencoder-based representations. Feature-level fusion integrates complementary information before classification. Experimental evaluation demonstrates that AUDRON effectively differentiates drone acoustic signatures from background noise, achieving high accuracy while maintaining generalizability across varying conditions. AUDRON achieves 98.51 percent and 97.11 percent accuracy in binary and multiclass classification. The results highlight the advantage of combining multiple feature representations with deep learning for reliable acoustic drone detection, suggesting the framework's potential for deployment in security and surveillance applications where visual or radar sensing may be limited.

---

## 11. 论文ID: 2512.25061v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.25061v1.json'

---

## 12. 论文ID: 2512.24917v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.24917v1.json'

---

## 13. 论文ID: 2512.24901v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.24901v1.json'

---

## 14. 论文ID: 2512.24665v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.24665v1.json'

---

## 15. 论文ID: 2512.24643v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.24643v1.json'

---

## 16. A Graph Neural Network with Auxiliary Task Learning for Missing PMU Data Reconstruction

**论文链接:** [http://arxiv.org/abs/2512.24542v1](http://arxiv.org/abs/2512.24542v1)

**作者:** Bo Li, Zijun Chen, Haiwang Zhong, Di Cao, Guangchun Ruan

**发布时间:** 2025-12-31

### GPT解析

### 总结

本文提出了一种辅助任务学习方法用于重建广域测量系统中缺失的相量测量单元数据，解决了现有方法在概念漂移、高缺失率和不完全可观测性方面的局限性。

### 背景

在广域测量系统中，相量测量单元数据容易因硬件故障、通信延迟和网络攻击而缺失。现有数据驱动方法存在不适应电力系统概念漂移、高缺失率下鲁棒性差以及依赖系统完全可观测性假设等问题。

### 目的

开发一种能够有效重建缺失PMU数据的方法，克服现有方法的局限性，提高在高缺失率和系统不完全可观测条件下的性能。

### 方法

提出了一种基于辅助任务学习的方法，包括：1) K跳图神经网络，允许在PMU节点子图上直接学习；2) 由两个互补图网络组成的辅助学习框架，包括时空GNN提取时空依赖关系重建缺失值，以及辅助GNN利用PMU数据低秩特性实现无监督在线学习。

### 主要发现

数值结果表明，所提出的方法在高缺失率和不完全可观测条件下展现出优越的离线和在线性能。

### 结论

辅助任务学习方法能够有效解决PMU数据缺失问题，通过动态利用PMU数据的低秩特性确保了方法的鲁棒性和自适应性。

### 翻译

在广域测量系统中，由于硬件故障、通信延迟和网络攻击，相量测量单元的测量数据容易缺失。现有的数据驱动方法受限于对电力系统中概念漂移的不适应性、高缺失率下的鲁棒性差，以及对系统完全可观测性的不切实际假设的依赖。因此，本文提出了一种用于重建缺失PMU数据的辅助任务学习方法。首先，提出了一种K跳图神经网络，使能够在由PMU节点组成的子图上进行直接学习，克服了系统不完全可观测的限制。然后，设计了一个由两个互补图网络组成的辅助学习框架，用于准确重建：一个时空GNN从PMU数据中提取时空依赖关系来重建缺失值，另一个辅助GNN利用PMU数据的低秩特性实现无监督在线学习。通过这种方式，PMU数据的低秩特性在整个架构中被动态利用，确保了鲁棒性和自适应性。数值结果表明，在高缺失率和不完全可观测条件下，所提出的方法具有优越的离线和在线性能。


### 论文摘要

In wide-area measurement systems (WAMS), phasor measurement unit (PMU) measurement is prone to data missingness due to hardware failures, communication delays, and cyber-attacks. Existing data-driven methods are limited by inadaptability to concept drift in power systems, poor robustness under high missing rates, and reliance on the unrealistic assumption of full system observability. Thus, this paper proposes an auxiliary task learning (ATL) method for reconstructing missing PMU data. First, a K-hop graph neural network (GNN) is proposed to enable direct learning on the subgraph consisting of PMU nodes, overcoming the limitation of the incompletely observable system. Then, an auxiliary learning framework consisting of two complementary graph networks is designed for accurate reconstruction: a spatial-temporal GNN extracts spatial-temporal dependencies from PMU data to reconstruct missing values, and another auxiliary GNN utilizes the low-rank property of PMU data to achieve unsupervised online learning. In this way, the low-rank properties of the PMU data are dynamically leveraged across the architecture to ensure robustness and self-adaptation. Numerical results demonstrate the superior offline and online performance of the proposed method under high missing rates and incomplete observability.

---

## 17. Networked Markets, Fragmented Data: Adaptive Graph Learning for Customer Risk Analytics and Policy Design

**论文链接:** [http://arxiv.org/abs/2512.24487v1](http://arxiv.org/abs/2512.24487v1)

**作者:** Lecheng Zheng, Jian Ni, Chris Zobel, John R Birge

**发布时间:** 2025-12-30

### GPT解析

### 总结

本文开发了一个集成客户智能框架，结合联邦学习、关系网络分析和自适应目标策略，帮助金融机构在保护客户数据隐私的同时，有效识别高风险客户行为并优化干预策略。

### 背景

金融机构在识别大规模交易网络中的高风险客户行为时面临挑战，欺诈活动利用市场分割和机构边界，而现有系统面临数据孤岛、行为类别极度不平衡以及客户干预策略不优化等问题。

### 目的

解决客户风险分析中的三个基本问题：数据孤岛阻碍全面关系评估、极端行为类别不平衡、以及无法平衡合规成本与关系价值的次优客户干预策略。

### 方法

开发一个集成的客户智能框架，结合联邦学习、关系网络分析和自适应目标策略；使用联邦图神经网络实现跨机构协作行为建模；引入跨银行个性化PageRank识别协调行为集群；使用分层强化学习机制优化动态干预目标。

### 主要发现

分析七个市场的140万笔客户交易后，该方法将误报率和漏报率分别降至4.64%和11.07%，显著优于单一机构模型；框架可预防79.25%的潜在损失，而固定规则政策仅能预防49.41%；最优市场特定目标阈值反映了异质客户群体特征。

### 结论

联邦客户分析显著提高了网络化竞争市场中的风险管理效果和客户关系成果。

### 翻译

金融机构在识别大规模交易网络中高风险客户行为方面面临日益严峻的挑战，其中欺诈活动利用市场分割和机构边界。我们解决了客户风险分析中的三个基本问题：阻碍全面关系评估的数据孤岛、极端的行为类别不平衡，以及无法平衡合规成本与关系价值的次优客户干预策略。我们开发了一个集成的客户智能框架，结合了联邦学习、关系网络分析和自适应目标策略。我们的联邦图神经网络使竞争机构能够协作建模行为，同时不损害专有客户数据，使用隐私保护嵌入来捕获跨市场关系模式。我们引入跨银行个性化PageRank来识别协调的行为集群，为风险管理人员提供可解释的客户网络细分。分层强化学习机制优化动态干预目标，校准升级政策以最大化预防价值，同时最小化客户摩擦和运营成本。分析七个市场的140万笔客户交易后，我们的方法将误报率和漏报率分别降至4.64%和11.07%，显著优于单一机构模型。该框架预防了79.25%的潜在损失，而固定规则政策下为49.41%，最佳市场特定目标阈值反映了异质客户群体特征。这些发现表明，在联网的竞争市场中，联邦客户分析显著改善了风险管理效果和客户关系成果。


### 论文摘要

Financial institutions face escalating challenges in identifying high-risk customer behaviors within massive transaction networks, where fraudulent activities exploit market fragmentation and institutional boundaries. We address three fundamental problems in customer risk analytics: data silos preventing holistic relationship assessment, extreme behavioral class imbalance, and suboptimal customer intervention strategies that fail to balance compliance costs with relationship value. We develop an integrated customer intelligence framework combining federated learning, relational network analysis, and adaptive targeting policies. Our federated graph neural network enables collaborative behavior modeling across competing institutions without compromising proprietary customer data, using privacy-preserving embeddings to capture cross-market relational patterns. We introduce cross-bank Personalized PageRank to identify coordinated behavioral clusters providing interpretable customer network segmentation for risk managers. A hierarchical reinforcement learning mechanism optimizes dynamic intervention targeting, calibrating escalation policies to maximize prevention value while minimizing customer friction and operational costs. Analyzing 1.4 million customer transactions across seven markets, our approach reduces false positive and false negative rates to 4.64% and 11.07%, substantially outperforming single-institution models. The framework prevents 79.25% of potential losses versus 49.41% under fixed-rule policies, with optimal market-specific targeting thresholds reflecting heterogeneous customer base characteristics. These findings demonstrate that federated customer analytics materially improve both risk management effectiveness and customer relationship outcomes in networked competitive markets.

---

## 18. Physics-informed Graph Neural Networks for Operational Flood Modeling

**论文链接:** [http://arxiv.org/abs/2512.23964v1](http://arxiv.org/abs/2512.23964v1)

**作者:** Carlo Malapad Acosta, Herath Mudiyanselage Viraj Vidura Herath, Jia Yu Lim, Abhishek Saha, Sanka Rasnayaka, Lucy Marshall

**发布时间:** 2025-12-30

**备注:** To be submitted to IJCAI

### GPT解析

### 总结

本文提出了一种名为DUALFloodGNN的新型洪水图神经网络架构，通过在全局和局部尺度嵌入物理约束，实现了对多个水文变量的高效准确预测。

### 背景

洪水模型通过模拟洪水的时空水动力学支持战略灾害管理，但基于物理的数值洪水模型计算成本高，限制了其在需要快速预测的操作环境中的应用。图神经网络(GNNs)设计的模型既能提供速度又能保持准确性，同时能够处理非结构化空间域。

### 目的

开发一种新的洪水GNN架构，能够在全局和局部尺度上嵌入物理约束，提高洪水预测的效率和准确性。

### 方法

提出了DUALFloodGNN架构，通过显式损失项在全局和局部尺度嵌入物理约束，并通过共享消息传递框架联合预测节点处的水体积和沿边的流量。为提高自回归推理性能，使用多步损失与动态课程学习相结合进行模型训练。

### 主要发现

与标准GNN架构和最先进的GNN洪水模型相比，DUALFloodGNN在预测多个水文变量方面取得了显著改进，同时保持了高计算效率。

### 结论

DUALFloodGNN模型已在https://github.com/acostacos/dual_flood_gnn开源，为洪水预测提供了一种高效准确的新方法。

### 翻译

洪水模型通过模拟洪水的时空水动力学为战略灾害管理提供信息。虽然基于物理的数值洪水模型准确，但其巨大的计算成本限制了它们在需要快速预测的操作环境中的使用。使用图神经网络(GNNs)设计的模型既能提供速度和准确性，又能处理非结构化空间域。鉴于其灵活的输入和架构，GNNs可以轻松地与物理感知技术结合使用，显著提高可解释性。本研究引入了一种新的洪水GNN架构DUALFloodGNN，它通过显式损失项在全局和局部尺度嵌入物理约束。该模型通过共享消息传递框架联合预测节点处的水体积和沿边的流量。为了提高自回归推理的性能，模型训练采用多步损失与动态课程学习相结合的方式进行。与标准GNN架构和最先进的GNN洪水模型相比，DUALFloodGNN在预测多个水文变量方面取得了显著改进，同时保持了高计算效率。该模型已在https://github.com/acostacos/dual_flood_gnn开源。


### 论文摘要

Flood models inform strategic disaster management by simulating the spatiotemporal hydrodynamics of flooding. While physics-based numerical flood models are accurate, their substantial computational cost limits their use in operational settings where rapid predictions are essential. Models designed with graph neural networks (GNNs) provide both speed and accuracy while having the ability to process unstructured spatial domains. Given its flexible input and architecture, GNNs can be leveraged alongside physics-informed techniques with ease, significantly improving interpretability. This study introduces a novel flood GNN architecture, DUALFloodGNN, which embeds physical constraints at both global and local scales through explicit loss terms. The model jointly predicts water volume at nodes and flow along edges through a shared message-passing framework. To improve performance for autoregressive inference, model training is conducted with a multi-step loss enhanced with dynamic curriculum learning. Compared with standard GNN architectures and state-of-the-art GNN flood models, DUALFloodGNN achieves substantial improvements in predicting multiple hydrologic variables while maintaining high computational efficiency. The model is open-sourced at https://github.com/acostacos/dual_flood_gnn.

---

## 19. Hardware Acceleration for Neural Networks: A Comprehensive Survey

**论文链接:** [http://arxiv.org/abs/2512.23914v1](http://arxiv.org/abs/2512.23914v1)

**作者:** Bin Xu, Ayan Banerjee, Sandeep Gupta

**发布时间:** 2025-12-30

### GPT解析

### 总结

这篇综述审查了深度学习硬件加速的技术景观，分析了当前面临的挑战并指出了未来发展方向。

### 背景

神经网络已成为云和边缘平台的主要计算负载，但模型规模和部署多样性的快速增长暴露了硬件瓶颈，这些瓶颈越来越受内存移动、通信和非规则算子的限制，而非峰值算术吞吐量。

### 目的

系统梳理深度学习硬件加速的技术现状，包括各类加速器架构和优化方法，并识别未来研究方向。

### 方法

使用统一的分类法组织技术空间，包括工作负载（CNN、RNN、GNN和Transformers/LLMs）、执行环境（训练与推理；数据中心与边缘）和优化杠杆（降低精度、稀疏化和剪枝、算子融合、编译和调度、内存系统/互连设计）。

### 主要发现

综合了关键架构思想，包括脉动阵列、向量和SIMD引擎、专用注意力和softmax内核、量化感知数据路径和高带宽内存，软件栈和编译器在连接模型语义与硬件方面发挥重要作用。

### 结论

开放挑战包括高效的长上下文LLM推理、对动态和稀疏工作负载的稳健支持、能源和感知安全的部署以及公平的基准测试，下一代神经加速器需要解决这些挑战。

### 翻译

神经网络已成为云和边缘平台的主要计算负载，但模型规模和部署多样性的快速增长暴露了硬件瓶颈，这些瓶颈越来越受内存移动、通信和非规则算子的限制，而非峰值算术吞吐量。这篇综述审查了深度学习硬件加速的技术景观，包括GPU和张量核心架构、领域特定加速器（如TPU/NPU）、基于FPGA的设计、ASIC推理引擎以及新兴的LLM服务加速器（如LPUs），同时还包括内存内/近内存计算和神经形态/模拟方法。我们使用统一的分类法组织技术空间，包括工作负载（CNN、RNN、GNN和Transformers/LLMs）、执行环境（训练与推理；数据中心与边缘）和优化杠杆（降低精度、稀疏化和剪枝、算子融合、编译和调度、内存系统/互连设计）。我们综合了关键架构思想，包括脉动阵列、向量和SIMD引擎、专用注意力和softmax内核、量化感知数据路径和高带宽内存，并讨论了软件栈和编译器如何将模型语义与硬件连接起来。最后，我们指出了开放挑战——包括高效的长上下文LLM推理（KV缓存管理）、对动态和稀疏工作负载的稳健支持、能源和感知安全的部署以及公平的基准测试——并指出了下一代神经加速的有前途的方向。


### 论文摘要

Neural networks have become a dominant computational workload across cloud and edge platforms, but rapid growth in model size and deployment diversity has exposed hardware bottlenecks increasingly dominated by memory movement, communication, and irregular operators rather than peak arithmetic throughput. This survey reviews the technology landscape for hardware acceleration of deep learning, spanning GPUs and tensor-core architectures; domain-specific accelerators (e.g., TPUs/NPUs); FPGA-based designs; ASIC inference engines; and emerging LLM-serving accelerators such as LPUs (language processing units), alongside in-/near-memory computing and neuromorphic/analog approaches. We organize the space using a unified taxonomy across (i) workloads (CNNs, RNNs, GNNs, and Transformers/LLMs), (ii) execution settings (training vs.\ inference; datacenter vs.\ edge), and (iii) optimization levers (reduced precision, sparsity and pruning, operator fusion, compilation and scheduling, and memory-system/interconnect design). We synthesize key architectural ideas including systolic arrays, vector and SIMD engines, specialized attention and softmax kernels, quantization-aware datapaths, and high-bandwidth memory, and we discuss how software stacks and compilers bridge model semantics to hardware. Finally, we highlight open challenges -- including efficient long-context LLM inference (KV-cache management), robust support for dynamic and sparse workloads, energy- and security-aware deployment, and fair benchmarking -- and point to promising directions for the next generation of neural acceleration.

---

## 20. Quantum Error Mitigation with Attention Graph Transformers for Burgers Equation Solvers on NISQ Hardware

**论文链接:** [http://arxiv.org/abs/2512.23817v1](http://arxiv.org/abs/2512.23817v1)

**作者:** Seyed Mohamad Ali Tousi, Adib Bazgir, Yuwen Zhang, G. N. DeSouza

**发布时间:** 2025-12-29

### GPT解析

### 总结

该研究提出了一种结合学习错误缓解的混合量子-经典框架，用于在NISQ硬件上求解粘性Burgers方程。通过Cole-Hopf变换和量子态编码，研究团队实现了对Burgers方程的量子模拟，并利用图神经网络进行错误缓解，显著提高了量子计算结果的准确性。

### 背景

当前量子计算技术处于NISQ时代，量子计算机存在噪声问题，限制了复杂问题的求解能力。Burgers方程作为流体力学中的基本方程，其量子求解面临非线性处理和噪声挑战。传统的零噪声外推方法在错误缓解方面存在局限性，需要更先进的解决方案。

### 目的

开发一种能够在NISQ硬件上准确求解粘性Burgers方程的混合量子-经典框架，并通过机器学习方法实现更有效的错误缓解，提高量子计算结果的准确性和可靠性。

### 方法

1. 使用Cole-Hopf变换将非线性Burgers方程转换为线性扩散方程；2. 在均匀网格上离散化方程并编码到量子态；3. 通过Qiskit实现Trotter化最近邻电路进行量子时间演化；4. 在噪声Aer后端和IBM量子设备上执行模拟；5. 构建大型参数化数据集，包含各种参数配置下的解；6. 训练基于注意力的图神经网络，利用电路结构和噪声输出预测错误缓解解。

### 主要发现

1. 学习的错误缓解方法在广泛的参数范围内持续减少了量子解和经典解之间的差异；2. 该方法超越了传统零噪声外推技术的效果；3. 基于注意力的图神经网络能够有效结合电路结构、光锥信息和全局参数进行预测。

### 结论

学习的错误缓解方法作为基于物理的噪声减少技术的补充，在NISQ设备上具有广阔的应用前景。该方法不仅可以应用于Burgers方程，还可以扩展到更高维的Burgers系统和更一般的量子偏微分方程求解器，为量子计算在科学计算中的应用提供了新的可能性。

### 翻译

我们提出了一种结合学习错误缓解的混合量子-经典框架，用于在噪声中等规模量子硬件上求解粘性Burgers方程。使用Cole-Hopf变换，将非线性Burgers方程映射到扩散方程，在均匀网格上离散化，并编码到量子态中，其时间演化通过Qiskit中实现的Trotter化最近邻电路来近似。量子模拟在噪声Aer后端和IBM超导量子设备上执行，并与使用基于Krylov的求解器应用于相应离散哈密顿量获得的高精度经典解进行基准测试。从测量的量子振幅中，我们重构速度场并评估物理和数值诊断，包括L2误差、激波位置和耗散率，有和无零噪声外推的情况。为实现数据驱动的错误缓解，我们通过扫描粘度、时间步长、网格分辨率和边界条件，构建了一个大型参数化数据集，产生匹配的噪声、ZNE校正、硬件和经典解的元组，以及详细的电路元数据。利用该数据集，我们训练了一个基于注意力的图神经网络，该网络结合了电路结构、光锥信息、全局电路参数和噪声量子输出以预测错误缓解的解决方案。在广泛的参数范围内，学习模型持续减少了量子解和经典解之间的差异，超过了单独使用ZNE所达到的效果。我们讨论了这种方法对更高维Burgers系统和更一般的量子偏微分方程求解器的扩展，强调了在NISQ设备上，学习的错误缓解作为一种基于物理的噪声减少技术的有前途的补充。


### 论文摘要

We present a hybrid quantum-classical framework augmented with learned error mitigation for solving the viscous Burgers equation on noisy intermediate-scale quantum (NISQ) hardware. Using the Cole-Hopf transformation, the nonlinear Burgers equation is mapped to a diffusion equation, discretized on uniform grids, and encoded into a quantum state whose time evolution is approximated via Trotterized nearest-neighbor circuits implemented in Qiskit. Quantum simulations are executed on noisy Aer backends and IBM superconducting quantum devices and are benchmarked against high-accuracy classical solutions obtained using a Krylov-based solver applied to the corresponding discretized Hamiltonian. From measured quantum amplitudes, we reconstruct the velocity field and evaluate physical and numerical diagnostics, including the L2 error, shock location, and dissipation rate, both with and without zero-noise extrapolation (ZNE). To enable data-driven error mitigation, we construct a large parametric dataset by sweeping viscosity, time step, grid resolution, and boundary conditions, producing matched tuples of noisy, ZNE-corrected, hardware, and classical solutions together with detailed circuit metadata. Leveraging this dataset, we train an attention-based graph neural network that incorporates circuit structure, light-cone information, global circuit parameters, and noisy quantum outputs to predict error-mitigated solutions. Across a wide range of parameters, the learned model consistently reduces the discrepancy between quantum and classical solutions beyond what is achieved by ZNE alone. We discuss extensions of this approach to higher-dimensional Burgers systems and more general quantum partial differential equation solvers, highlighting learned error mitigation as a promising complement to physics-based noise reduction techniques on NISQ devices.

---

## 21. A Survey on Graph Neural Networks for Fraud Detection in Ride Hailing Platforms

**论文链接:** [http://arxiv.org/abs/2512.23777v1](http://arxiv.org/abs/2512.23777v1)

**作者:** Kanishka Hewageegana, Janani Harischandra, Nipuna Senanayake, Gihan Danansuriya, Kavindu Hapuarachchi, Pooja Illangarathne

**发布时间:** 2025-12-29

**DOI:** 10.1109/ICAIBD62003.2024.10604597

**备注:** 12 pages, 8 figures, 2 tables. Presented at the 2024 7th International Conference on Artificial Intelligence and Big Data (ICAIBD)

### GPT解析

### 总结

该研究探讨了使用图神经网络(GNNs)检测网约车平台欺诈行为，比较了不同模型的有效性，分析了常见欺诈活动，解决了类别不平衡和欺诈伪装问题，并概述了GNN架构在异常检测中的应用。

### 背景

网约车平台存在欺诈行为问题，需要有效的检测方法。现有的欺诈检测工作存在局限性，且面临类别不平衡和欺诈伪装等挑战。

### 目的

研究使用图神经网络检测网约车平台欺诈行为的有效性，分析常见欺诈活动，比较现有工作，解决类别不平衡和欺诈伪装问题，并探索GNN架构在异常检测中的应用。

### 方法

分析常见的欺诈活动，比较现有的欺诈检测工作，概述GNN架构和方法在异常检测中的应用，识别方法进展和差距。

### 主要发现

确定了GNN在欺诈检测中的方法进展和差距，发现需要解决类别不平衡和欺诈伪装问题。

### 结论

需要进一步探索图神经网络在网约车欺诈检测中的实际应用和技术改进，以增强快速发展的网约车行业中的欺诈检测策略。

### 翻译

本研究通过图神经网络(GNNs)调查网约车平台中的欺诈检测，重点关注各种模型的有效性。通过分析普遍存在的欺诈活动，研究强调了与欺诈检测相关的现有工作，这对于解决网约车平台内的欺诈事件非常有用。此外，论文还强调了处理类别不平衡和欺诈伪装的问题。它概述了应用于异常检测的GNN架构和方法的系统性概述，确定了重要的方法进展和差距。论文呼吁进一步探索实际应用可行性和技术改进，以增强快速发展的网约车行业中的欺诈检测策略。


### 论文摘要

This study investigates fraud detection in ride hailing platforms through Graph Neural Networks (GNNs),focusing on the effectiveness of various models. By analyzing prevalent fraudulent activities, the research highlights and compares the existing work related to fraud detection which can be useful when addressing fraudulent incidents within the online ride hailing platforms. Also, the paper highlights addressing class imbalance and fraudulent camouflage. It also outlines a structured overview of GNN architectures and methodologies applied to anomaly detection, identifying significant methodological progress and gaps. The paper calls for further exploration into real-world applicability and technical improvements to enhance fraud detection strategies in the rapidly evolving ride-hailing industry.

---

## 22. 论文ID: 2512.24793v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.24793v1.json'

---

## 23. UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2512.24763v1](http://arxiv.org/abs/2512.24763v1)

**作者:** Ankit Dhiman, Srinath R, Jaswanth Reddy, Lokesh R Boregowda, Venkatesh Babu Radhakrishnan

**发布时间:** 2025-12-31

**备注:** Accepted to AAAI 2026. Project Page: https://unic-lift.github.io/

### GPT解析

### 总结

本文提出了一种统一框架，将3D高斯飞溅(3DGS)和神经辐射场(NeRF)技术应用于新视角合成，解决了多视角2D分割到3D转换中的标签一致性问题，并通过可学习特征嵌入和边界困难样本挖掘技术提高了分割性能。

### 背景

3D高斯飞溅(3DGS)和神经辐射场(NeRF)技术已推动了新视角合成的发展，最近的方法将多视角2D分割扩展到3D以实现更好的场景理解，但面临不同视角下2D实例标签不一致的挑战。

### 目的

开发一个统一框架，合并现有的两阶段方法，减少训练时间并提高性能；引入可学习特征嵌入用于高斯原语分割；解决对象边界处的伪影问题。

### 方法

提出统一框架合并现有步骤；引入可学习特征嵌入用于高斯原语分割；通过'嵌入到标签'过程解码实例标签；针对对象边界问题实施困难样本挖掘；在计算三元组损失前对光栅化特征嵌入应用线性层以稳定训练。

### 主要发现

统一框架虽提供显著优势但在对象边界处出现伪影；边界困难样本挖掘有助于解决此问题；特征嵌入上应用线性层稳定了训练并显著提高了性能；在ScanNet、Replica3D和Messy-Rooms数据集上质量和数量均优于基线方法。

### 结论

所提出的方法在多个数据集上表现出色，证明了其在新视角合成和3D分割任务中的有效性和实用性。

### 翻译

三维高斯飞溅和神经辐射场已推动了新视角合成的发展。最近的方法将多视角二维分割扩展到三维，实现了实例/语义分割以更好地理解场景。一个关键挑战是不同视角下二维实例标签的不一致性，导致三维预测效果不佳。现有方法采用两阶段方法，一些依赖于超参数敏感的聚类的对比学习，而其他方法则预处理标签以实现一致性。我们提出一个统一框架，合并这些步骤，通过引入高斯原语分割的可学习特征嵌入来减少训练时间并提高性能。然后通过新颖的'嵌入到标签'过程将此特征有效地解码为实例标签，有效集成了优化。虽然这个统一框架提供了显著优势，但我们在对象边界处观察到伪影。为解决对象边界问题，我们提出在这些边界上进行困难样本挖掘。然而，直接将困难挖掘应用于特征嵌入被证明是不稳定的。因此，我们在计算三元组损失之前对光栅化的特征嵌入应用线性层，这稳定了训练并显著提高了性能。我们的方法在ScanNet、Replica3D和Messy-Rooms数据集上在质量和数量上都优于基线方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D实例分割中的多视图2D标签不一致问题。当使用2D分割方法处理多视角图像时，不同视角下的分割标签可能不一致，导致3D预测效果差。这个问题在现实中非常重要，因为AR/VR、自动驾驶、路径规划等应用都需要对3D场景有准确的理解，而现有方法要么需要昂贵的后处理步骤，要么训练时间过长，效率低下。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有方法的不足：两阶段方法计算成本高，训练时间长；对比学习方法需要昂贵的后处理步骤；特征蒸馏方法训练时间过长。基于这些分析，作者设计了统一的单阶段框架，借鉴了3D高斯溅射(3DGS)作为基础表示方法，以及Contrastive-Lift的对比学习思想，但避免了其高维特征蒸馏的复杂性和昂贵的后处理步骤。作者还提出了专门的边界困难样本挖掘策略和'嵌入到标签'过程来解决对象边界问题和直接生成标签。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是将2D分割标签提升到3D表示，解决多视图标签不一致问题，使用统一的单阶段框架直接从可学习的3D嵌入生成一致的分割标签，并通过对比学习和三元组损失优化嵌入空间。整体实现流程包括：1)初始化3D高斯原语的向量嵌入；2)渲染向量嵌入到相机视图；3)应用对比损失优化嵌入；4)应用三元组损失处理边界样本；5)3D邻域正则化确保空间一致性；6)通过'嵌入到标签'过程解码生成最终分割结果。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)统一的单阶段框架，合并特征学习和标签解码；2)'嵌入到标签'过程，通过简单阈值操作直接生成标签，避免昂贵聚类；3)边界困难样本挖掘策略，提高对象边界分割质量；4)高效的训练和推理，将训练时间从20小时以上减少到40分钟以内。相比之前的工作，UniC-Lift消除了传统两阶段方法中的预处理或后处理步骤，避免了Contrastive-Lift所需的HDBSCAN聚类后处理，以及特征蒸馏方法的高计算复杂度和长时间训练问题。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': "UniC-Lift通过统一的单阶段框架和创新的'嵌入到标签'过程，显著提高了3D实例分割的效率和准确性，解决了多视图2D标签不一致的问题，同时大幅减少了训练时间。"}


### 论文摘要

3D Gaussian Splatting (3DGS) and Neural Radiance Fields (NeRF) have advanced novel-view synthesis. Recent methods extend multi-view 2D segmentation to 3D, enabling instance/semantic segmentation for better scene understanding. A key challenge is the inconsistency of 2D instance labels across views, leading to poor 3D predictions. Existing methods use a two-stage approach in which some rely on contrastive learning with hyperparameter-sensitive clustering, while others preprocess labels for consistency. We propose a unified framework that merges these steps, reducing training time and improving performance by introducing a learnable feature embedding for segmentation in Gaussian primitives. This embedding is then efficiently decoded into instance labels through a novel "Embedding-to-Label" process, effectively integrating the optimization. While this unified framework offers substantial benefits, we observed artifacts at the object boundaries. To address the object boundary issues, we propose hard-mining samples along these boundaries. However, directly applying hard mining to the feature embeddings proved unstable. Therefore, we apply a linear layer to the rasterized feature embeddings before calculating the triplet loss, which stabilizes training and significantly improves performance. Our method outperforms baselines qualitatively and quantitatively on the ScanNet, Replica3D, and Messy-Rooms datasets.

---

## 24. Automated Classification of First-Trimester Fetal Heart Views Using Ultrasound-Specific Self-Supervised Learning

**论文链接:** [http://arxiv.org/abs/2512.24492v1](http://arxiv.org/abs/2512.24492v1)

**作者:** Youssef Megahed, Aylin Erman, Robin Ducharme, Mark C. Walker, Steven Hawken, Adrian D. C. Chan

**发布时间:** 2025-12-30

**备注:** 7 pages, 4 figures

### GPT解析

### 总结

该研究评估了一种名为USF-MAE的自监督超声基础模型，用于早期胎儿心脏视图分类。该模型在大量未标记的超声图像上预训练后，在分类任务中表现出色，准确率达到90.57%，优于其他基线模型。

### 背景

先天性心脏病是最常见的先天性异常，也是新生儿发病和死亡的主要原因。早期胎儿超声心动图虽可提供更早检测机会，但因心脏结构小、信噪比低及操作者间变异性大，使自动化分析面临挑战。

### 目的

评估USF-MAE自监督超声基础模型在早期胎儿心脏视图分类任务中的性能。

### 方法

USF-MAE在超过37万张未标记的超声图像上使用掩码自编码模型进行预训练，涵盖40多个解剖区域。随后在6,720张早期胎儿超声心动图图像数据集上对预训练的Vision Transformer编码器进行微调，用于分类五个类别：主动脉、房室血流、V征、X征和其他。与监督卷积神经网络基线和自然图像预训练的Vision Transformer模型进行性能对比。

### 主要发现

在独立测试集上，USF-MAE在所有评估指标上表现最佳，准确率达90.57%，精确率91.15%，召回率90.57%，F1分数90.71%。与最强基线ResNet-18相比，准确率提高2.03%，F1分数提高1.98%。

### 结论

该方法在不依赖激进图像预处理或感兴趣区域裁剪的情况下表现出稳健性能，并改善了对非诊断帧的鉴别能力。

### 翻译

先天性心脏病仍然是最常见的先天性异常，也是新生儿发病和死亡的主要原因。尽管早期胎儿超声心动图为更早的检测提供了机会，但由于心脏结构小、信噪比低以及操作者间变异性大，这一阶段的自动化分析具有挑战性。在这项工作中，我们评估了一种名为USF-MAE的自监督超声基础模型，用于早期胎儿心脏视图分类。USF-MAE使用掩码自编码模型在超过37万张未标记的超声图像上进行预训练，这些图像跨越40多个解剖区域，随后在下游分类任务中进行微调。作为概念验证，预训练的Vision Transformer编码器在一个开源的6,720张早期胎儿超声心动图图像数据集上进行了微调，用于分类五个类别：主动脉、房室血流、V征、X征和其他。模型性能与监督卷积神经网络基线（ResNet-18和ResNet-50）以及在自然图像（ImageNet-1k）上预训练的Vision Transformer（ViT-B/16）模型进行了基准测试。所有模型都使用相同的预处理、数据分割和优化协议进行训练和评估。在独立测试集上，USF-MAE在所有评估指标上实现了最佳性能，准确率达到90.57%，精确率为91.15%，召回率为90.57%，F1得分为90.71%。与最强的基线ResNet-18相比，这代表了准确率提高+2.03%，F1分数提高+1.98%。所提出的方法在不依赖激进图像预处理或感兴趣区域裁剪的情况下表现出稳健的性能，并显示了对非诊断帧的改进鉴别能力。


### 论文摘要

Congenital heart disease remains the most common congenital anomaly and a leading cause of neonatal morbidity and mortality. Although first-trimester fetal echocardiography offers an opportunity for earlier detection, automated analysis at this stage is challenging due to small cardiac structures, low signal-to-noise ratio, and substantial inter-operator variability. In this work, we evaluate a self-supervised ultrasound foundation model, USF-MAE, for first-trimester fetal heart view classification. USF-MAE is pretrained using masked autoencoding modelling on more than 370,000 unlabelled ultrasound images spanning over 40 anatomical regions and is subsequently fine-tuned for downstream classification. As a proof of concept, the pretrained Vision Transformer encoder was fine-tuned on an open-source dataset of 6,720 first-trimester fetal echocardiography images to classify five categories: aorta, atrioventricular flows, V sign, X sign, and Other. Model performance was benchmarked against supervised convolutional neural network baselines (ResNet-18 and ResNet-50) and a Vision Transformer (ViT-B/16) model pretrained on natural images (ImageNet-1k). All models were trained and evaluated using identical preprocessing, data splits, and optimization protocols. On an independent test set, USF-MAE achieved the highest performance across all evaluation metrics, with 90.57% accuracy, 91.15% precision, 90.57% recall, and 90.71% F1-score. This represents an improvement of +2.03% in accuracy and +1.98% in F1-score compared with the strongest baseline, ResNet-18. The proposed approach demonstrated robust performance without reliance on aggressive image preprocessing or region-of-interest cropping and showed improved discrimination of non-diagnostic frames.

---

## 25. Lifting Vision: Ground to Aerial Localization with Reasoning Guided Planning

**论文链接:** [http://arxiv.org/abs/2512.24404v1](http://arxiv.org/abs/2512.24404v1)

**作者:** Soham Pahari, M. Srinivas

**发布时间:** 2025-12-30

### GPT解析

### 总结

本文提出了一种名为ViReLoc的视觉推理框架，用于仅通过视觉表示进行规划和定位，解决了传统推理系统在空间任务中的局限性。

### 背景

多模态智能在视觉理解和高级推理方面进展显著，但大多数推理系统仍主要依赖文本信息进行推理，限制了它们在视觉导航和地理定位等空间任务中的有效性。

### 目的

讨论该领域的潜在范围，并提出Geo-Consistent Visual视觉推理范式及ViReLoc框架，实现仅使用视觉表示进行规划和定位。

### 方法

ViReLoc框架通过在视觉域中逐步编码推理并使用基于强化的目标进行优化，学习空间依赖性和几何关系；整合对比学习和自适应特征交互，对齐跨视角并减少视角差异。

### 主要发现

在各种导航和定位场景中的实验显示，空间推理准确性和跨视角检索性能一致提高；视觉推理可作为导航和定位的强大补充方法，无需实时GPS数据即可执行任务。

### 结论

视觉推理是导航和定位的有效补充方法，可在没有实时全球定位系统数据的情况下执行空间任务，从而实现更安全的导航解决方案。

### 翻译

多模态智能发展最近在视觉理解和高级推理方面显示出强劲进展。然而，大多数推理系统仍然主要依赖文本信息作为推理的主要媒介。这限制了它们在视觉导航和地理定位等空间任务中的有效性。本文讨论了该领域的潜在范围，并最终提出了一种视觉推理范式Geo-Consistent Visual Planning，我们引入的框架称为Visual Reasoning for Localization，或ViReLoc，它仅使用视觉表示进行规划和定位。所提出的框架学习空间依赖性和几何关系，而基于文本的推理往往难以理解这些关系。通过在视觉域中逐步编码推理并使用基于强化的目标进行优化，ViReLoc在两个给定的地面图像之间规划路线。该系统还整合了对比学习和自适应特征交互，以对齐跨视角并减少视角差异。在各种导航和定位场景中的实验显示，空间推理准确性和跨视角检索性能一致提高。这些结果确立了视觉推理作为导航和定位的强大补充方法，并表明此类任务可以在没有实时全球定位系统数据的情况下执行，从而实现更安全的导航解决方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何实现从地面视角到空中视角的地理定位（Ground-to-Aerial Localization），并在此过程中引入视觉推理引导的规划。这个问题很重要，因为它解决了传统系统依赖文本信息和GPS数据的局限性，使导航系统在GPS信号不可用或不可靠的环境中仍能工作，同时提供更安全、可解释的导航解决方案。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别了现有方法的局限性：大多数多模态系统主要依赖文本推理，造成视觉输入与文本推理间的差距；传统地理定位系统仅进行检索而不进行环境理解和规划。作者借鉴了视觉规划中的强化学习思想、跨视角地理定位中的特征提取和对比学习方法，以及大型视觉模型中的视觉链生成方法。在此基础上，作者创新性地设计了将地理定位视为视觉推理任务的统一框架，整合了跨视角编码、视觉推理、地图构建和导航规划。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将跨视角地理定位视为逐步的视觉推理问题，通过生成中间视觉状态连接地面和空中视图，构建可导航地图用于规划。整体流程包括三阶段：1)画布构建：创建包含卫星图像和拓扑图的地理空间支架；2)跨视角地理定位：使用学生-教师架构对齐地面和空中特征；3)视觉规划：通过强化学习生成轨迹，结合A*算法和地理一致性奖励函数进行路径规划和优化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的架构整合跨视角编码、视觉推理和导航规划；2)视觉推理模块生成中间状态 bridging 地面和空中视图；3)可微分规划系统使用联合训练和奖励信号构建自由空间；4)完全使用视觉表示，不依赖文本或GPS数据；5)使用对比学习和自适应特征交互减少视角差异。相比之前工作，ViReLoc将定位视为推理而非检索问题，生成中间视觉状态表示视角转换，并通过强化学习连接特征、推理和规划，实现了端到端的地理一致性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ViReLoc框架通过纯视觉推理实现了从地面到空间的地理定位和导航规划，为无需GPS数据的安全、可解释导航系统开辟了新方向。'}


### 论文摘要

Multimodal intelligence development recently show strong progress in visual understanding and high level reasoning. Though, most reasoning system still reply on textual information as the main medium for inference. This limit their effectiveness in spatial tasks such as visual navigation and geo-localization. This work discuss about the potential scope of this field and eventually propose an idea visual reasoning paradigm Geo-Consistent Visual Planning, our introduced framework called Visual Reasoning for Localization, or ViReLoc, which performs planning and localization using only visual representations. The proposed framework learns spatial dependencies and geometric relations that text based reasoning often suffer to understand. By encoding step by step inference in the visual domain and optimizing with reinforcement based objectives, ViReLoc plans routes between two given ground images. The system also integrates contrastive learning and adaptive feature interaction to align cross view perspectives and reduce viewpoint differences. Experiments across diverse navigation and localization scenarios show consistent improvements in spatial reasoning accuracy and cross view retrieval performance. These results establish visual reasoning as a strong complementary approach for navigation and localization, and show that such tasks can be performed without real time global positioning system data, leading to more secure navigation solutions.

---

## 26. Skim-Aware Contrastive Learning for Efficient Document Representation

**论文链接:** [http://arxiv.org/abs/2512.24373v1](http://arxiv.org/abs/2512.24373v1)

**作者:** Waheed Ahmed Abro, Zied Bouraoui

**发布时间:** 2025-12-30

### GPT解析

### 总结

本研究提出了一种新的自监督对比学习框架，通过模拟人类阅读策略来增强长文档表示，在法律和医学领域取得了显著效果。

### 背景

基于transformer的模型在词和句子级任务中表现出色，但在表示长文档（特别是在法律和医学领域）方面仍然存在困难。稀疏注意力机制资源密集且无法捕捉完整文档上下文，分层transformer模型虽效率更高但未明确说明如何关联文档不同部分。

### 目的

开发一种能够有效表示长文档的方法，特别是针对法律和医学等领域的文本，同时提高计算效率和文档表示的丰富性。

### 方法

引入一种新的自监督对比学习框架，随机屏蔽文档的一部分，并使用基于自然语言推理(NLI)的对比目标，使其与相关部分对齐，同时与不相关部分分离，模拟人类综合信息的过程。

### 主要发现

在法律和生物医学文本上的实验证实，该方法在文档表示的准确性和计算效率方面都有显著提升。

### 结论

这种模拟人类阅读策略的自监督对比学习方法能够有效解决长文档表示的挑战，特别是在法律和医学等专业领域，提供了更丰富且计算效率更高的文档表示。

### 翻译

虽然基于transformer的模型在词和句子级任务中表现出色，但有效地表示长文档，特别是在法律和医学等领域，仍然很困难。稀疏注意力机制可以处理更长的输入，但资源密集且往往无法捕捉完整的文档上下文。分层transformer模型提供了更好的效率，但没有明确说明它们如何关联文档的不同部分。相比之下，人类通常浏览文本，关注重要部分以理解整体信息。受此人类策略启发，我们引入了一种新的自监督对比学习框架，以增强长文档表示。我们的方法随机屏蔽文档的一部分，并使用基于自然语言推理(NLI)的对比目标，使其与相关部分对齐，同时与不相关部分分离。这模拟了人类如何综合信息，产生了更丰富且计算效率更高的表示。在法律和生物医学文本上的实验证实了在准确性和效率方面都有显著提升。


### 论文摘要

Although transformer-based models have shown strong performance in word- and sentence-level tasks, effectively representing long documents, especially in fields like law and medicine, remains difficult. Sparse attention mechanisms can handle longer inputs, but are resource-intensive and often fail to capture full-document context. Hierarchical transformer models offer better efficiency but do not clearly explain how they relate different sections of a document. In contrast, humans often skim texts, focusing on important sections to understand the overall message. Drawing from this human strategy, we introduce a new self-supervised contrastive learning framework that enhances long document representation. Our method randomly masks a section of the document and uses a natural language inference (NLI)-based contrastive objective to align it with relevant parts while distancing it from unrelated ones. This mimics how humans synthesize information, resulting in representations that are both richer and more computationally efficient. Experiments on legal and biomedical texts confirm significant gains in both accuracy and efficiency.

---

## 27. Empower Low-Altitude Economy: A Reliability-Aware Dynamic Weighting Allocation for Multi-modal UAV Beam Prediction

**论文链接:** [http://arxiv.org/abs/2512.24324v1](http://arxiv.org/abs/2512.24324v1)

**作者:** Haojin Li, Anbang Zhang, Chen Sun, Chenyuan Feng, Kaiqian Qu, Tony Q. S. Quek, Haijun Zhang

**发布时间:** 2025-12-30

### GPT解析

### 总结

本文提出了一种名为SaM2B的语义感知多模态波束预测框架，通过可靠性感知的动态加权方案解决无人机通信中波束预测问题，提高了在复杂环境下的连接可靠性。

### 背景

低空经济由城市空中交通、物流无人机和空中感知驱动快速发展，无人机通信中快速准确的波束预测对实现可靠连接至关重要，当前研究正从单一信号转向多模态协同方法。

### 目的

解决现有多模态方法使用固定权重导致的问题，提出一种能够根据不同场景动态调整模态权重的波束预测框架，提高跨场景泛化能力。

### 方法

设计了SaM2B框架，利用环境视觉、飞行姿态和地理空间数据等轻量级线索，通过可靠性感知的动态权重更新自适应分配各模态贡献，并利用跨模态对比学习将多源表示波束语义对齐到共享语义空间，增强判别能力和鲁棒性。

### 主要发现

在真实低空无人机数据集上的实验表明，SaM2B比基线方法取得了更满意的结果，有效解决了模态不匹配和弱对齐问题，提高了波束预测的准确性和鲁棒性。

### 结论

SaM2B框架通过动态权重调整和语义对齐，能够有效处理多模态数据在不同场景下的可靠性变化，显著提升了无人机通信中的波束预测性能和连接可靠性。

### 翻译

低空经济(LAE)由城市空中交通、物流无人机和空中感知驱动而迅速扩张，同时，在无人机通信中进行快速准确的波束预测对于实现可靠连接至关重要。当前研究正从单一信号转向多模态协同方法。然而，现有的多模态方法大多采用固定或经验权重，假设在任何给定时刻所有模态的可靠性都相等。实际上，不同模态的重要性随着无人机运动场景的变化而大幅波动，静态权重放大了退化模态的负面影响。此外，模态不匹配和弱对齐进一步削弱了跨场景泛化能力。为此，我们提出了一种应用于语义感知多模态波束预测框架的可靠性感知动态加权方案，命名为SaM2B。具体而言，SaM2B利用环境视觉、飞行姿态和地理空间数据等轻量级线索，通过可靠性感知的动态权重更新，在不同时间点自适应地分配各模态的贡献。此外，通过利用跨模态对比学习，我们将与特定波束信息相关的'多源表示波束语义'对齐到共享语义空间，从而在模态噪声和分布变化下增强判别能力和鲁棒性。在真实低空无人机数据集上的实验表明，SaM2B比基线方法取得了更满意的结果。


### 论文摘要

The low-altitude economy (LAE) is rapidly expanding driven by urban air mobility, logistics drones, and aerial sensing, while fast and accurate beam prediction in uncrewed aerial vehicles (UAVs) communications is crucial for achieving reliable connectivity. Current research is shifting from single-signal to multi-modal collaborative approaches. However, existing multi-modal methods mostly employ fixed or empirical weights, assuming equal reliability across modalities at any given moment. Indeed, the importance of different modalities fluctuates dramatically with UAV motion scenarios, and static weighting amplifies the negative impact of degraded modalities. Furthermore, modal mismatch and weak alignment further undermine cross-scenario generalization. To this end, we propose a reliability-aware dynamic weighting scheme applied to a semantic-aware multi-modal beam prediction framework, named SaM2B. Specifically, SaM2B leverages lightweight cues such as environmental visual, flight posture, and geospatial data to adaptively allocate contributions across modalities at different time points through reliability-aware dynamic weight updates. Moreover, by utilizing cross-modal contrastive learning, we align the "multi-source representation beam semantics" associated with specific beam information to a shared semantic space, thereby enhancing discriminative power and robustness under modal noise and distribution shifts. Experiments on real-world low-altitude UAV datasets show that SaM2B achieves more satisfactory results than baseline methods.

---

## 28. Balanced Hierarchical Contrastive Learning with Decoupled Queries for Fine-grained Object Detection in Remote Sensing Images

**论文链接:** [http://arxiv.org/abs/2512.24074v1](http://arxiv.org/abs/2512.24074v1)

**作者:** Jingzhou Chen, Dexin Chen, Fengchao Xiong, Yuntao Qian, Liang Xiao

**发布时间:** 2025-12-30

### GPT解析

### 总结

本文提出了一种平衡的层次对比损失和解耦学习策略，用于解决细粒度遥感目标检测中的挑战，特别是在处理分层标签结构和数据分布不均衡问题上。

### 背景

细粒度遥感数据集通常使用分层标签结构以粗到细的方式区分对象，每个对象在多个层次上都有标注。将这种语义层次结构嵌入到表示学习空间以提高细粒度检测性能仍然具有挑战性。

### 目的

解决细粒度遥感检测中两个关键问题：(1)标签层次结构中数据分布不均衡导致高频类主导学习过程，(2)类别间语义关系的学习干扰了与类别无关的定位。

### 方法

提出了一种平衡的层次对比损失，结合检测变压器(DETR)框架中的解耦学习策略。该损失引入可学习的类别原型，平衡不同层次各类别对梯度的贡献；解耦策略将DETR的对象查询分为分类和定位集合，实现任务特定的特征提取和优化。

### 主要发现

平衡的层次对比损失确保每个层次类别在每个小批量中对损失计算的贡献相等；解耦策略使分类和定位任务能够独立优化，提高了模型性能。

### 结论

在三个具有分层标注的细粒度数据集上的实验表明，所提出的方法优于最先进的方法，有效解决了细粒度遥感目标检测中的挑战。

### 翻译

细粒度遥感数据集通常使用分层标签结构以粗到细的方式区分对象，每个对象在多个层次上都有标注。然而，将这种语义层次结构嵌入到表示学习空间以提高细粒度检测性能仍然具有挑战性。先前的研究在不同层次上应用监督对比学习，将同一父类下的对象分组，同时区分兄弟子类别。然而，他们忽略了两个关键问题：(1)标签层次结构中数据分布不均衡导致高频类主导学习过程，(2)类别间语义关系的学习干扰了与类别无关的定位。为解决这些问题，我们提出了一种平衡的层次对比损失，结合检测变压器(DETR)框架中的解耦学习策略。所提出的损失引入可学习的类别原型，平衡不同层次各类别对梯度的贡献，确保每个层次类别在每个小批量中对损失计算的贡献相等。解耦策略将DETR的对象查询分为分类和定位集合，实现任务特定的特征提取和优化。在三个具有分层标注的细粒度数据集上的实验表明，我们的方法优于最先进的方法。


### 论文摘要

Fine-grained remote sensing datasets often use hierarchical label structures to differentiate objects in a coarse-to-fine manner, with each object annotated across multiple levels. However, embedding this semantic hierarchy into the representation learning space to improve fine-grained detection performance remains challenging. Previous studies have applied supervised contrastive learning at different hierarchical levels to group objects under the same parent class while distinguishing sibling subcategories. Nevertheless, they overlook two critical issues: (1) imbalanced data distribution across the label hierarchy causes high-frequency classes to dominate the learning process, and (2) learning semantic relationships among categories interferes with class-agnostic localization. To address these issues, we propose a balanced hierarchical contrastive loss combined with a decoupled learning strategy within the detection transformer (DETR) framework. The proposed loss introduces learnable class prototypes and equilibrates gradients contributed by different classes at each hierarchical level, ensuring that each hierarchical class contributes equally to the loss computation in every mini-batch. The decoupled strategy separates DETR's object queries into classification and localization sets, enabling task-specific feature extraction and optimization. Experiments on three fine-grained datasets with hierarchical annotations demonstrate that our method outperforms state-of-the-art approaches.

---

## 29. Tracing the Heart's Pathways: ECG Representation Learning from a Cardiac Conduction Perspective

**论文链接:** [http://arxiv.org/abs/2512.24002v1](http://arxiv.org/abs/2512.24002v1)

**作者:** Tan Pan, Yixuan Sun, Chen Jiang, Qiong Gao, Rui Sun, Xingmeng Zhang, Zhenqi Yang, Limei Han, Yixiu Liang, Yuan Cheng, Kaiyu Guo

**发布时间:** 2025-12-30

**备注:** Accepted to AAAI2026

### GPT解析

### 总结

本文提出CLEAR-HUG两阶段框架，通过捕捉导联间心脏传导的细微变化并遵循心电图诊断指南，显著提高了心电图分析性能。

### 背景

多导联心电图是心脏诊断的基础，近期心电图自监督学习虽无需高质量标注，但现有方法专注于导联和心搏间的一致性模式，忽视了心脏传导过程中固有的心搏差异，且未遵循从心搏到导联组合的诊断指南逻辑。

### 目的

解决现有eSSL方法忽视心脏传导细微变化的问题，使表征学习与心电图诊断指南的渐进逻辑保持一致。

### 方法

提出CLEAR-HUG两阶段框架：第一阶段使用传导-导联重建器(CLEAR)模型，将每个心搏视为独立实体，采用稀疏注意机制捕捉心搏间特定变化和一般共性；第二阶段实现分层导联统一组头(HUG)用于疾病诊断，模拟临床工作流程。

### 主要发现

在六项任务上的实验结果显示CLEAR-HUG提高了6.84%，验证了其有效性和捕捉心脏传导细微变化的能力。

### 结论

CLEAR-HUG能够增强心脏传导的表征能力，并将模式与专家诊断指南保持一致，在心电图分析领域具有重要应用价值。

### 翻译

多导联心电图(ECG)是心脏诊断的基石。近期心电图自监督学习(eSSL)的进展为增强表征学习提供了前景，无需依赖高质量标注。然而，早期的eSSL方法存在一个关键局限：它们专注于导联和心搏间的一致性模式，忽视了心脏传导过程中固有的心搏差异，而这些细微但有重要意义的变异携带独特的生理特征。此外，ECG分析的表征学习应符合ECG诊断指南，该指南从单个心搏到单导联，最终到导联组合是渐进式的。然而，在将预训练模型应用于下游任务时，这种顺序逻辑常被忽略。为解决这些差距，我们提出了CLEAR-HUG，一个两阶段框架，旨在捕捉导联间心脏传导的细微变化，同时遵循ECG诊断指南。在第一阶段，我们引入了一个名为传导-导联重建器(CLEAR)的eSSL模型，捕捉心搏间特定变化和一般共性。将每个心搏视为独立实体，CLEAR采用简单但有效的稀疏注意机制重建信号，避免其他心搏干扰。在第二阶段，我们实现了分层导联统一组头(HUG)用于疾病诊断，模拟临床工作流程。在六项任务上的实验结果显示提高了6.84%，验证了CLEAR-HUG的有效性。这突显了其增强心脏传导表征能力并将模式与专家诊断指南保持一致的能力。


### 论文摘要

The multi-lead electrocardiogram (ECG) stands as a cornerstone of cardiac diagnosis. Recent strides in electrocardiogram self-supervised learning (eSSL) have brightened prospects for enhancing representation learning without relying on high-quality annotations. Yet earlier eSSL methods suffer a key limitation: they focus on consistent patterns across leads and beats, overlooking the inherent differences in heartbeats rooted in cardiac conduction processes, while subtle but significant variations carry unique physiological signatures. Moreover, representation learning for ECG analysis should align with ECG diagnostic guidelines, which progress from individual heartbeats to single leads and ultimately to lead combinations. This sequential logic, however, is often neglected when applying pre-trained models to downstream tasks. To address these gaps, we propose CLEAR-HUG, a two-stage framework designed to capture subtle variations in cardiac conduction across leads while adhering to ECG diagnostic guidelines. In the first stage, we introduce an eSSL model termed Conduction-LEAd Reconstructor (CLEAR), which captures both specific variations and general commonalities across heartbeats. Treating each heartbeat as a distinct entity, CLEAR employs a simple yet effective sparse attention mechanism to reconstruct signals without interference from other heartbeats. In the second stage, we implement a Hierarchical lead-Unified Group head (HUG) for disease diagnosis, mirroring clinical workflow. Experimental results across six tasks show a 6.84% improvement, validating the effectiveness of CLEAR-HUG. This highlights its ability to enhance representations of cardiac conduction and align patterns with expert diagnostic guidelines.

---

## 30. Wireless Multimodal Foundation Model (WMFM): Integrating Vision and Communication Modalities for 6G ISAC Systems

**论文链接:** [http://arxiv.org/abs/2512.23897v1](http://arxiv.org/abs/2512.23897v1)

**作者:** Mohammad Farzanullah, Han Zhang, Akram Bin Sediq, Ali Afana, Melike Erol-Kantarci

**发布时间:** 2025-12-29

**备注:** Journal Paper, 13 pages, 11 figures, 4 tables

### GPT解析

### 总结

该研究提出了一种基于对比学习的无线多模态基础模型（WMFM），能够同时从无线信道系数和视觉图像中学习，通过预训练编码器作为特征提取器，并使用轻量级任务特定头部进行微调，显著提升了用户定位和LoS/nLoS分类任务的性能，同时大幅减少了训练时间。

### 背景

多模态基础模型的兴起通过实现跨多种数据类型的联合理解，革新了学习范式。在下一代无线网络中，集成感知和通信模态为开发可泛化和数据高效的模型提供了独特机会。

### 目的

开发一种能够同时处理无线信道数据和视觉数据的大规模框架，以提高在用户定位和LoS/nLoS分类等任务上的性能，同时减少训练时间和数据需求。

### 方法

作者提出了基于对比学习的无线多模态基础模型（WMFM），使用对比学习进行预训练，这是一种自监督学习技术，能够在不需要显式标签的情况下对齐摄像头和信道数据的嵌入。预训练后的编码器被冻结并用作特征提取器，然后使用轻量级的特定任务头部进行微调。

### 主要发现

在DeepVerse6G数据集上的实验表明，WMFM与端到端基准相比，在LoS/nLoS分类中实现了17%的平衡准确率提升，定位误差减少了48.5%，同时将训练时间减少了最多90倍。即使仅使用20%的数据进行训练，基于WMFM的头部也优于完全监督的端到端模型，突显了其鲁棒性和数据高效学习能力。

### 结论

所提出的方法为集成感知和通信系统中的可扩展多模态学习奠定了基础，为智能和自适应的6G网络铺平了道路。

### 翻译

多模态基础模型的出现通过实现跨多种数据类型的联合理解，革新了学习范式。在下一代无线网络的背景下，集成感知和通信模态为开发可泛化和数据高效的模型提供了独特机会。在这项工作中，我们引入了基于对比学习的无线多模态基础模型（WMFM），这是一个大规模框架，能够同时从无线信道系数和视觉图像中学习。WMFM使用对比学习进行预训练，这是一种自监督学习技术，能够在不需要显式标签的情况下对齐摄像头和信道数据的嵌入。然后，预训练的编码器被冻结并用作特征提取器，使用轻量级的特定任务头部进行微调，用于下游任务，包括用户定位和LoS/nLoS分类。在DeepVerse6G数据集上的大量实验表明，所提出的WMFM与端到端基准相比，在LoS/nLoS分类中实现了17%的平衡准确率提升，定位误差减少了48.5%，同时将训练时间减少了最多90倍。即使仅使用20%的数据进行训练，基于WMFM的头部也优于完全监督的端到端模型，突显了它们的鲁棒性和数据高效学习能力。所提出的方法为集成感知和通信系统中的可扩展多模态学习奠定了基础，为智能和自适应的6G网络铺平了道路。


### 论文摘要

The emergence of multimodal foundation models has revolutionized learning paradigms by enabling joint understanding across diverse data types. In the context of next-generation wireless networks, integrating sensing and communication modalities presents a unique opportunity to develop generalizable and data-efficient models. In this work, we introduce the contrastive learning based Wireless Multimodal Foundation Model (WMFM), a large-scale framework that jointly learns from wireless channel coefficients and visual imagery. The WMFM is pretrained using contrastive learning, a self-supervised learning technique that aligns embeddings of camera and channel data without requiring explicit labels. The pretrained encoders are then frozen and employed as feature extractors, with lightweight task-specific heads, fine-tuned for downstream tasks, including user localization and LoS/nLoS classification. Extensive experiments on the DeepVerse6G dataset demonstrate that the proposed WMFM achieves a 17% improvement in balanced accuracy for LoS/nLoS classification and a 48.5% reduction in localization error compared to the end-to-end (E2E) benchmark, while reducing training time by up to 90-fold. Even when trained with as little as 20% of the data, the WMFM-based heads outperform the fully supervised E2E model, underscoring their robustness and data-efficient learning. The proposed approach establishes a foundation for scalable, multimodal learning in Integrated Sensing and Communication (ISAC) systems, paving the way for intelligent and adaptive 6G networks.

---

## 31. HINTS: Extraction of Human Insights from Time-Series Without External Sources

**论文链接:** [http://arxiv.org/abs/2512.23755v1](http://arxiv.org/abs/2512.23755v1)

**作者:** Sheo Yon Jhin, Noseong Park

**发布时间:** 2025-12-27

**备注:** AAAI 2026 AI4TS Workshop paper

### GPT解析

### 总结

该研究提出了HINTS框架，一种自监督学习方法，无需外部数据即可从时间序列残差中提取影响金融和经济系统的人类因素。

### 背景

人类决策、情感和集体心理是影响金融和经济系统时间动态性的复杂因素。现有时间序列预测模型虽利用外部数据源捕捉这些因素，但存在高数据依赖成本。

### 目的

开发一种不依赖外部数据的时间序列预测方法，能够内生提取影响金融和经济系统的人类因素。

### 方法

提出HINTS框架，利用Friedkin-Johnsen意见动态模型作为结构归纳偏置来建模社会影响、记忆和偏见模式，并将提取的因素整合到骨干模型作为注意力图。

### 主要发现

在九个真实世界和基准数据集上的实验表明，HINTS持续提高预测准确性；案例研究和消融研究验证了其可解释性，提取的因素与真实世界事件有强语义一致性。

### 结论

HINTS框架在时间序列预测中具有实际应用价值，能有效捕捉影响金融和经济系统的人类因素。

### 翻译

人类决策、情感和集体心理是塑造金融和经济系统中观察到的时间动态性的复杂因素。最近的时间序列预测模型利用外部来源（如新闻和社交媒体）来捕捉人类因素，但这些方法在财务、计算和实际含义方面产生高数据依赖成本。在本研究中，我们提出了HINTS，一种自监督学习框架，无需外部数据即可从时间序列残差中内生提取这些潜在因素。HINTS利用Friedkin-Johnsen(FJ)意见动态模型作为结构归纳偏置来建模不断发展的社会影响、记忆和偏见模式。提取的人类因素被整合到最先进的骨干模型中作为注意力图。使用九个真实世界和基准数据集进行的实验结果表明，HINTS持续提高了预测准确性。此外，多个案例研究和消融研究验证了HINTS的可解释性，表明提取的因素与真实世界事件之间存在强烈的语义一致性，证明了HINTS的实际应用价值。


### 论文摘要

Human decision-making, emotions, and collective psychology are complex factors that shape the temporal dynamics observed in financial and economic systems. Many recent time series forecasting models leverage external sources (e.g., news and social media) to capture human factors, but these approaches incur high data dependency costs in terms of financial, computational, and practical implications. In this study, we propose HINTS, a self-supervised learning framework that extracts these latent factors endogenously from time series residuals without external data. HINTS leverages the Friedkin-Johnsen (FJ) opinion dynamics model as a structural inductive bias to model evolving social influence, memory, and bias patterns. The extracted human factors are integrated into a state-of-the-art backbone model as an attention map. Experimental results using nine real-world and benchmark datasets demonstrate that HINTS consistently improves forecasting accuracy. Furthermore, multiple case studies and ablation studies validate the interpretability of HINTS, demonstrating strong semantic alignment between the extracted factors and real-world events, demonstrating the practical utility of HINTS.

---

## 32. FoundationSLAM: Unleashing the Power of Depth Foundation Models for End-to-End Dense Visual SLAM

**论文链接:** [http://arxiv.org/abs/2512.25008v1](http://arxiv.org/abs/2512.25008v1)

**作者:** Yuchen Wu, Jiahe Li, Fabio Tosi, Matteo Poggi, Jin Zheng, Xiao Bai

**发布时间:** 2025-12-31

### GPT解析

### 总结

论文提出了FoundationSLAM，一种基于学习的单目密集SLAM系统，通过结合流估计与几何推理解决了先前方法中缺乏几何一致性的问题，实现了准确且鲁棒的跟踪和映射。

### 背景

之前的基于流的方法在单目SLAM中缺乏几何一致性，需要一种能够准确且鲁棒地进行跟踪和映射的新方法。

### 目的

开发一种能够解决几何一致性问题的单目密集SLAM系统，实现准确且鲁棒的跟踪和映射。

### 方法

提出FoundationSLAM系统，包含混合流网络生成具有几何感知能力的对应关系，双一致性束调整层在多视图约束下联合优化关键帧姿态和深度，以及可靠性感知细化机制通过区分可靠和不确定区域动态调整流更新过程，形成匹配与优化之间的闭环反馈。

### 主要发现

FoundationSLAM在多个具有挑战性的数据集上实现了优越的轨迹精度和密集重建质量，系统能够以18 FPS的速度实时运行，对各种场景表现出强大的泛化能力。

### 结论

FoundationSLAM是一种有效的单目密集SLAM系统，解决了先前方法中的几何一致性问题，具有实用性和广泛的适用场景。

### 翻译

我们提出了FoundationSLAM，一种基于学习的单目密集SLAM系统，它解决了先前基于流的方法中缺乏几何一致性的问题，以实现准确且鲁棒的跟踪和映射。我们的核心思想是通过利用基础深度模型的指导，将流估计与几何推理相结合。为此，我们首先开发了一个混合流网络，它产生具有几何感知能力的对应关系，使不同关键帧之间能够进行一致的深度和姿态推断。为了强制执行全局一致性，我们提出了一个双一致性束调整层，在多视图约束下联合优化关键帧姿态和深度。此外，我们引入了一种可靠性感知的细化机制，通过区分可靠和不确定区域来动态调整流更新过程，形成匹配和优化之间的闭环反馈。大量实验表明，FoundationSLAM在多个具有挑战性的数据集上实现了优越的轨迹精度和密集重建质量，同时以18 FPS的速度实时运行，展示了我们对各种场景的强大泛化能力和我们方法的实用性。


### 论文摘要

We present FoundationSLAM, a learning-based monocular dense SLAM system that addresses the absence of geometric consistency in previous flow-based approaches for accurate and robust tracking and mapping. Our core idea is to bridge flow estimation with geometric reasoning by leveraging the guidance from foundation depth models. To this end, we first develop a Hybrid Flow Network that produces geometry-aware correspondences, enabling consistent depth and pose inference across diverse keyframes. To enforce global consistency, we propose a Bi-Consistent Bundle Adjustment Layer that jointly optimizes keyframe pose and depth under multi-view constraints. Furthermore, we introduce a Reliability-Aware Refinement mechanism that dynamically adapts the flow update process by distinguishing between reliable and uncertain regions, forming a closed feedback loop between matching and optimization. Extensive experiments demonstrate that FoundationSLAM achieves superior trajectory accuracy and dense reconstruction quality across multiple challenging datasets, while running in real-time at 18 FPS, demonstrating strong generalization to various scenarios and practical applicability of our method.

---

## 33. ArtiSG: Functional 3D Scene Graph Construction via Human-demonstrated Articulated Objects Manipulation

**论文链接:** [http://arxiv.org/abs/2512.24845v1](http://arxiv.org/abs/2512.24845v1)

**作者:** Qiuyi Gu, Yuze Sheng, Jincheng Yu, Jiahao Tang, Xiaolong Shan, Zhaoyang Shen, Tinghao Yi, Xiaodan Liang, Xinlei Chen, Yu Wang

**发布时间:** 2025-12-31

### GPT解析

### 总结

ArtiSG是一个通过将人类演示编码为结构化机器人记忆来构建功能性3D场景图的框架，能够准确估计铰接对象的运动轨迹和轴，发现视觉感知错过的功能元素，并在真实环境中有效指导机器人执行语言指导的操作任务。

### 背景

3D场景图虽为机器人提供了语义理解能力用于导航和规划，但缺乏物理操作所需的函数信息，特别是铰接对象的信息。现有从静态观察推断铰接机制的方法易受视觉模糊影响，而从状态变化估计参数的方法依赖受限设置。通用目标检测器常错过精细功能元素如小把手。

### 目的

弥补现有3D场景图在物理操作功能信息方面的不足，特别是针对铰接对象，构建包含功能信息的3D场景图。

### 方法

提出ArtiSG框架，通过将人类演示编码为结构化机器人记忆来构建功能性3D场景图。利用便携式设置的数据收集流程准确估计6自由度铰接轨迹和轴，整合运动学先验到分层开放词汇图中，并利用交互数据发现视觉感知错过的功能元素。

### 主要发现

真实世界实验表明，ArtiSG在功能元素召回率和铰接估计精度方面显著优于基线方法。构建的场景图可作为可靠功能记忆，有效指导机器人在包含各种铰接对象的真实环境中执行语言指导的操作任务。

### 结论

ArtiSG框架能够有效构建包含铰接对象信息的功能性3D场景图，作为机器人的功能记忆，帮助其在复杂环境中执行操作任务。

### 翻译

三维场景图已赋予机器人语义理解能力，用于导航和规划，但它们通常缺乏物理操作所需的函数信息，特别是关于铰接对象的信息。从静态观察推断铰接机制的现有方法容易受到视觉模糊的影响，而从状态变化估计参数的方法通常依赖于固定摄像头和无遮挡视图等受限设置。此外，通用目标检测器经常错过精细的功能元素，如小把手。为了弥补这一差距，我们提出了ArtiSG，一个通过将人类演示编码为结构化机器人记忆来构建功能性三维场景图的框架。我们的方法利用健壮的铰接数据收集流程，使用便携式设置来准确估计六自由度铰接轨迹和轴，即使在摄像头自运动的情况下。我们将这些运动学先验整合到分层和开放词汇图中，同时利用交互数据来发现视觉感知错过的不显眼功能元素。大量的真实世界实验证明，ArtiSG在功能元素召回率和铰接估计精度方面显著优于基线方法。此外，我们证明构建的图可作为可靠的功能记忆，有效地引导机器人在包含各种铰接对象的真实环境中执行语言指导的操作任务。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D场景图缺乏功能性信息的问题，特别是关于关节物体的操作机制。这个问题很重要，因为机器人需要在复杂环境中进行物理操作，而现有方法要么从静态图像推断关节机制（易受视觉歧义影响），要么依赖固定摄像头等约束条件（在现实世界难以保证），且容易忽略小功能元素（如把手）。这种功能性信息对机器人实现物理交互至关重要，能帮助机器人更自然地与人类环境互动。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受人类通过观察他人操作学习的启发，认为这种能力尚未被充分利用在机器人场景理解中。他们设计了一个框架，将人类操作编码为结构化场景图作为机器人记忆。借鉴了3D场景图（如Armeni的工作）、功能元素检测（如FunGraph、OpenFunGraph）、关节物体理解（如GFlow、RAM）和人类演示操作（如UMI、FastUMI）等现有工作，但创新性地将人类交互视为功能演示而非简单的对象跟踪，从而明确编码关节机制。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过人类演示的关节物体操作构建功能性3D场景图，将静态视觉感知与动态交互结合。流程分三阶段：1）功能性场景图初始化：使用3D分割提取物体，通过top-k帧选择最佳视角，检测功能元素并提取语义特征；2）视角鲁棒关节估计：使用便携式设备（头戴相机+带标记球的UMI夹爪）跟踪操作轨迹，估计关节轴和类型；3）交互增强图细化：将运动学信息与场景图关联，补充视觉检测遗漏的功能元素。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）功能性3D场景图构建框架，结合视觉模型和人类演示轨迹；2）视角鲁棒的数据收集管道，支持动态视角；3）交互增强的功能元素检测，发现视觉忽略的元素；4）开放词汇场景构建，提升泛化性。相比之前工作，ArtiSG不依赖固定视角，能处理视觉歧义，通过人类操作发现隐藏功能元素，并明确编码关节机制而非仅检测静态元素。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ArtiSG通过将人类演示的关节物体操作编码为功能性3D场景图，使机器人能够更准确地理解和操作复杂环境中的可关节物体，解决了现有场景图缺乏物理交互所需功能信息的问题。'}


### 论文摘要

3D scene graphs have empowered robots with semantic understanding for navigation and planning, yet they often lack the functional information required for physical manipulation, particularly regarding articulated objects. Existing approaches for inferring articulation mechanisms from static observations are prone to visual ambiguity, while methods that estimate parameters from state changes typically rely on constrained settings such as fixed cameras and unobstructed views. Furthermore, fine-grained functional elements like small handles are frequently missed by general object detectors. To bridge this gap, we present ArtiSG, a framework that constructs functional 3D scene graphs by encoding human demonstrations into structured robotic memory. Our approach leverages a robust articulation data collection pipeline utilizing a portable setup to accurately estimate 6-DoF articulation trajectories and axes even under camera ego-motion. We integrate these kinematic priors into a hierarchical and open-vocabulary graph while utilizing interaction data to discover inconspicuous functional elements missed by visual perception. Extensive real-world experiments demonstrate that ArtiSG significantly outperforms baselines in functional element recall and articulation estimation precision. Moreover, we show that the constructed graph serves as a reliable functional memory that effectively guides robots to perform language-directed manipulation tasks in real-world environments containing diverse articulated objects.

---

## 34. 3D Semantic Segmentation for Post-Disaster Assessment

**论文链接:** [http://arxiv.org/abs/2512.24593v1](http://arxiv.org/abs/2512.24593v1)

**作者:** Nhut Le, Maryam Rahnemoonfar

**发布时间:** 2025-12-31

**备注:** Accepted by the 2025 IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2025)

### GPT解析

### 总结

本研究针对自然灾害频发对人类生命和经济造成严重威胁的问题，构建了一个专门用于灾后评估的3D数据集，并评估了现有最先进的3D语义分割模型在该数据集上的表现，发现了现有方法的局限性，强调了开发专门技术和数据集的必要性。

### 背景

自然灾害发生频率增加，对人类生命构成严重威胁，并导致巨大经济损失。3D语义分割对灾后评估至关重要，但现有深度学习模型缺乏专门为灾后环境设计的数据集。

### 目的

解决现有模型缺乏专门数据集的问题，构建专门用于灾后评估的3D数据集，并评估最先进的3D语义分割模型在灾后环境中的表现。

### 方法

使用无人机捕获飓风伊恩影响地区的航拍影像，运用运动结构(SfM)和多视图立体(MVS)技术重建3D点云，构建专门的3D数据集，并评估Fast Point Transformer (FPT)、Point Transformer v3 (PTv3)和OA-CNNs等最先进的3D语义分割模型。

### 主要发现

现有最先进的3D语义分割模型在灾后地区存在显著局限性，无法有效处理灾后场景的特殊挑战。

### 结论

迫切需要改进3D分割技术并开发专门的3D基准数据集，以提高灾后场景理解和响应能力。

### 翻译

自然灾害发生频率的增加对人类生命构成严重威胁，并导致巨大的经济损失。虽然3D语义分割对灾后评估至关重要，但现有的深度学习模型缺乏专门为灾后环境设计的数据集。为解决这一差距，我们使用无人机在飓风伊恩影响区域捕获的航拍影像构建了一个专门的3D数据集，并采用运动结构(SfM)和多视图立体(MVS)技术重建3D点云。我们在这个数据集上评估了最先进的3D语义分割模型，包括快速点变换器(FPT)、点变换器v3(PTv3)和OA-CNNs，揭示了现有方法在受灾地区的显著局限性。这些发现强调了改进3D分割技术和开发专门3D基准数据集的迫切需求，以提高灾后场景理解和响应能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决缺乏专门用于灾后评估的3D语义分割数据集问题，以及现有的3D语义分割模型在灾后环境中表现不佳的问题。这个问题很重要，因为自然灾害频率增加，准确评估灾情对救援行动至关重要，而现有模型主要针对常规环境设计，无法有效处理灾后场景的特殊挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到现有3D数据集缺乏灾后场景专门设计，受RescueNet项目启发，收集飓风Ian受灾区域的无人机航拍影像，使用SfM和MVS技术重建3D点云，并通过人工标注生成真实标签。他们借鉴了RescueNet的数据收集和标注方法，以及成熟的3D重建技术，但没有重新发明算法，而是专注于创建专用数据集并评估现有模型。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建专门用于灾后评估的3D语义分割数据集，因为现有数据集无法满足灾后场景的特殊需求。整体流程包括：1)收集无人机航拍影像；2)图像预处理增强对比度；3)使用SfM和MVS技术重建3D点云；4)手动清理点云移除异常值；5)对每10帧图像进行标注并通过多数投票生成3D标签；6)使用3D编辑软件优化标签；7)划分数据集为训练、验证和测试集；8)评估三种先进模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)创建首个专门用于灾后评估的3D语义分割数据集；2)揭示现有最先进模型在灾后场景中的局限性；3)提供对灾后环境中3D语义分割挑战的深入理解。相比之前工作，本文数据集专注于灾后3D场景而非常规环境，包含受损和未受损建筑的区分；首次在灾后3D场景上评估先进模型；强调开发专门针对灾后场景的技术的必要性，而非通用场景理解。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过创建首个专门用于灾后评估的3D语义分割数据集并评估现有模型，揭示了现有方法在灾后场景中的局限性，为开发更适应灾后环境的3D语义分割技术奠定了基础。'}


### 论文摘要

The increasing frequency of natural disasters poses severe threats to human lives and leads to substantial economic losses. While 3D semantic segmentation is crucial for post-disaster assessment, existing deep learning models lack datasets specifically designed for post-disaster environments. To address this gap, we constructed a specialized 3D dataset using unmanned aerial vehicles (UAVs)-captured aerial footage of Hurricane Ian (2022) over affected areas, employing Structure-from-Motion (SfM) and Multi-View Stereo (MVS) techniques to reconstruct 3D point clouds. We evaluated the state-of-the-art (SOTA) 3D semantic segmentation models, Fast Point Transformer (FPT), Point Transformer v3 (PTv3), and OA-CNNs on this dataset, exposing significant limitations in existing methods for disaster-stricken regions. These findings underscore the urgent need for advancements in 3D segmentation techniques and the development of specialized 3D benchmark datasets to improve post-disaster scene understanding and response.

---

## 35. From Building Blocks to Planning: Multi-Step Spatial Reasoning in LLMs with Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2512.24532v1](http://arxiv.org/abs/2512.24532v1)

**作者:** Amir Tahmasbi, Sadegh Majidi, Kazem Taram, Aniket Bera

**发布时间:** 2025-12-31

### GPT解析

### 总结

本文提出了一种两阶段方法，用于提高大型语言模型的空间推理能力，将空间推理分解为基本构建块及其组合，并通过监督微调和轻量级LoRA适配器进行训练。

### 背景

大型语言模型在一般语言能力方面表现出色，但在结构化环境中的空间变换和多步骤规划方面仍然存在困难。空间推理在导航和规划应用中越来越受到关注。

### 目的

提高大型语言模型在空间变换和多步骤规划方面的能力，特别是在结构化环境中的表现。

### 方法

采用两阶段方法：第一阶段对基本空间变换（如旋转、平移和缩放）进行监督微调，使模型具备基本空间物理知识；第二阶段冻结具有物理知识的模型，在GRPO框架内训练轻量级LoRA适配器，学习在拼图环境中组合这些构建块进行多步骤规划的策略。同时合成了ASCII艺术数据集并构建了相应的强化学习环境。

### 主要发现

提出的方法在动态环境和静态环境中均优于基线模型（包括通用骨干模型、物理感知模型和端到端强化学习模型）；与从头开始的端到端强化学习相比，收敛更快且训练更稳定；通过注意力模式分析评估了微调对空间理解的改进效果。

### 结论

通过分解空间推理为基本构建块并采用两阶段训练方法，可以有效提高大型语言模型在空间推理任务上的性能，特别是在多步骤规划方面。

### 翻译

大型语言模型中的空间推理因其导航和规划应用而受到越来越多的关注。尽管具有强大的通用语言能力，LLMs在结构化环境中的空间变换和多步骤规划方面仍然存在困难。我们提出了一种两阶段方法，将空间推理分解为基本构建块及其组合。首先，我们在基本空间变换（如旋转、平移和缩放）上应用监督微调，使模型具备基本空间物理知识。然后，我们冻结这个具有物理感知能力的模型，并在GRPO框架内训练轻量级LoRA适配器，以学习在基于拼图的环境中组合这些构建块进行多步骤规划的策略，以闭环方式进行。为支持这一流程，我们合成了一个ASCII艺术数据集并构建了相应的基于ASCII的强化学习环境。我们的方法在动态环境（具有明确状态更新）和静态环境（模型必须依靠内部状态跨步骤）中均优于基线模型，包括通用骨干模型、物理感知模型和端到端强化学习模型。此外，与从头开始的端到端强化学习相比，所提出的方法收敛更快且表现出更稳定的训练。最后，我们分析了注意力模式，以评估微调是否在空间理解方面带来了有意义的改进。


### 论文摘要

Spatial reasoning in large language models (LLMs) has gained increasing attention due to applications in navigation and planning. Despite strong general language capabilities, LLMs still struggle with spatial transformations and multi-step planning in structured environments. We propose a two-stage approach that decomposes spatial reasoning into atomic building blocks and their composition. First, we apply supervised fine-tuning on elementary spatial transformations, such as rotation, translation, and scaling, to equip the model with basic spatial physics. We then freeze this physics-aware model and train lightweight LoRA adapters within the GRPO framework to learn policies that compose these building blocks for multi-step planning in puzzle-based environments, in a closed-loop manner. To support this pipeline, we synthesize an ASCII-art dataset and construct a corresponding ASCII-based reinforcement learning environment. Our method consistently outperforms baselines, including the generic backbone, physics-aware model, and end-to-end RL models, under both Dynamic environments with explicit state updates and Static environments where the model must rely on its internal state across steps. In addition, the proposed approach converges faster and exhibits more stable training compared to end-to-end reinforcement learning from scratch. Finally, we analyze attention patterns to assess whether fine-tuning induces meaningful improvements in spatial understanding.

---

## 36. Thinking on Maps: How Foundation Model Agents Explore, Remember, and Reason Map Environments

**论文链接:** [http://arxiv.org/abs/2512.24504v1](http://arxiv.org/abs/2512.24504v1)

**作者:** Zhiwei Wei, Yuxing Liu, Hua Liao, Wenjia Xu

**发布时间:** 2025-12-30

**备注:** 43 pages, 8 figures

### GPT解析

### 总结

该论文提出了一种交互式评估框架，分析基础模型代理人在符号地图环境中的探索、记忆和推理能力，揭示了探索策略、记忆表示和推理方案在空间理解中的不同功能角色。

### 背景

地图环境是表示空间结构的基础媒介，理解基础模型代理人在此类环境中的理解和行动方式对实现可靠的基于地图推理和应用至关重要。然而，现有评估大多依赖静态地图输入或文本查询，忽略了空间理解的交互性和经验驱动特性。

### 目的

提出一个交互式评估框架，分析基础模型代理人在符号地图环境中如何探索、记忆和推理，以更全面地评估其空间能力。

### 方法

让代理人逐步探索部分可观测的网格地图（包含道路、交叉点和兴趣点），每步仅接收局部观测。使用六种空间任务评估空间理解能力，通过系统地改变探索策略、记忆表示和推理方案，研究这些组件在不同基础模型中的功能角色。

### 主要发现

探索主要影响经验获取但对推理准确性影响有限；记忆表示在整合空间经验中起核心作用，结构化记忆显著提高结构密集型任务性能；推理方案影响如何利用存储的空间知识，高级提示支持更有效多步推理；空间推理性能在模型规模超过阈值后会趋于饱和。

### 结论

评估基础模型空间能力需考虑交互性和经验驱动特性；记忆表示和推理方案对空间理解至关重要；单纯扩大模型规模无法持续提升基于地图的空间推理能力，需要针对空间表示和推理的特定机制。

### 翻译

地图环境为表示空间结构提供了基础媒介。因此，理解基础模型(FM)代理人在此类环境中如何理解和行动，对于实现可靠的基于地图的推理和应用至关重要。然而，大多数对基础模型空间能力的现有评估依赖于静态地图输入或基于文本的查询，忽略了空间理解的交互性和经验驱动特性。在本文中，我们提出了一种交互式评估框架，用于分析基础模型代理人在符号地图环境中如何探索、记忆和推理。代理人逐步探索部分可观测的基于网格的地图，包含道路、交叉点和兴趣点(POIs)，每步仅接收局部观测。然后使用六种空间任务评估空间理解能力。通过在多个基础模型中系统地改变探索策略、记忆表示和推理方案，我们揭示了这些组件的不同功能角色。探索主要影响经验获取，但对最终推理准确性的影响有限。相比之下，记忆表示在整合空间经验中起核心作用，特别是结构化记忆（如顺序和基于图的表示）能显著提高结构密集型任务（如路径规划）的性能。推理方案进一步影响如何利用存储的空间知识，高级提示支持更有效的多步推理。我们还观察到，空间推理性能在模型版本和规模超过一定能力阈值后会趋于饱和，表明基于地图的空间理解的改进需要针对空间表示和推理的机制，而不仅仅是扩大规模。


### 论文摘要

Map environments provide a fundamental medium for representing spatial structure. Understanding how foundation model (FM) agents understand and act in such environments is therefore critical for enabling reliable map-based reasoning and applications. However, most existing evaluations of spatial ability in FMs rely on static map inputs or text-based queries, overlooking the interactive and experience-driven nature of spatial understanding.In this paper, we propose an interactive evaluation framework to analyze how FM agents explore, remember, and reason in symbolic map environments. Agents incrementally explore partially observable grid-based maps consisting of roads, intersections, and points of interest (POIs), receiving only local observations at each step. Spatial understanding is then evaluated using six kinds of spatial tasks. By systematically varying exploration strategies, memory representations, and reasoning schemes across multiple foundation models, we reveal distinct functional roles of these components. Exploration primarily affects experience acquisition but has a limited impact on final reasoning accuracy. In contrast, memory representation plays a central role in consolidating spatial experience, with structured memories particularly sequential and graph-based representations, substantially improving performance on structure-intensive tasks such as path planning. Reasoning schemes further shape how stored spatial knowledge is used, with advanced prompts supporting more effective multi-step inference. We further observe that spatial reasoning performance saturates across model versions and scales beyond a certain capability threshold, indicating that improvements in map-based spatial understanding require mechanisms tailored to spatial representation and reasoning rather than scaling alone.

---

## 37. Foundation models on the bridge: Semantic hazard detection and safety maneuvers for maritime autonomy with vision-language models

**论文链接:** [http://arxiv.org/abs/2512.24470v1](http://arxiv.org/abs/2512.24470v1)

**作者:** Kim Alexander Christensen, Andreas Gudahl Tufte, Alexey Gusev, Rohan Sinha, Milan Ganai, Ole Andreas Alsos, Marco Pavoned, Martin Steinert

**发布时间:** 2025-12-30

**备注:** 17 pages without bibliography or appendix. The main paper has 16 figures

### GPT解析

### 总结

该论文提出了一种名为'Semantic Lookout'的视觉-语言模型系统，用于海上自主船舶的备用操作选择，使其能够在检测到偏离操作设计域时采取适当行动，符合国际海事组织MASS法规草案的要求。

### 背景

国际海事组织(IMO)的MASS法规草案要求自主和远程监督的海上船舶能够检测偏离操作设计域的情况，进入预定义的备用模式，通知操作员，允许立即人工覆盖，并在未经批准的情况下不改变航行计划。传统的海上自主系统在正确行动取决于语义理解(如潜水员下潜标志意味着水中有人员，附近有火意味着危险)的情况下表现不佳。

### 目的

开发一种能够在人工接管窗口内提供语义感知的备用操作机制，使船舶能够在检测到异常情况时采取适当的短期、可人工覆盖的备用操作。

### 方法

研究提出了一种快速-慢速异常检测管道，结合短期、可人工覆盖的备用操作。他们引入了'Semantic Lookout'，一种仅基于摄像头、候选约束的视觉-语言模型(VLM)备用操作选择器，能够在持续人工授权下从水域有效、世界锚定的轨迹中选择一个谨慎的操作(或保持位置)。研究在40个港口场景中进行了测试，测量了场景理解和延迟、与人类共识的一致性、危险场景下的短期风险缓解以及端到端操作。

### 主要发现

10秒以下的模型保留了较慢最先进模型的大部分感知能力；备用操作选择器优于仅基于几何的基线方法，并在火灾场景中增加了 standoff 距离；现场运行验证了端到端操作。

### 结论

视觉-语言模型可作为符合国际海事组织MASS法规草案的语义备用操作选择器，在实际延迟预算内运行。这些发现支持未来在基础模型语义与多传感器鸟瞰感知和短期重新规划配对的领域自适应混合自主系统的研究。

### 翻译

国际海事组织MASS法规草案要求自主和远程监督的海上船舶检测偏离其操作设计域的情况，进入预定义的备用模式，通知操作员，允许立即人工覆盖，并在未经批准的情况下不改变航行计划。在警报到接管之间的间隙中满足这些要求需要短期、可人工覆盖的备用操作。传统的海上自主系统在正确行动取决于语义理解(如潜水员下潜标志意味着水中有人员，附近有火意味着危险)的情况下表现不佳。我们论证(i)视觉-语言模型(VLMs)为这类分布外情况提供了语义感知能力，以及(ii)具有短期、可人工覆盖备用操作的快速-慢速异常检测管道使得在人工接管窗口内实现这一功能成为可能。我们引入了Semantic Lookout，一种仅基于摄像头、候选约束的视觉-语言模型(VLM)备用操作选择器，在持续人工授权下从水域有效、世界锚定的轨迹中选择一个谨慎的操作(或保持位置)。在40个港口场景中，我们测量了每次调用的场景理解和延迟、与人类共识的一致性(模型三人多数投票)、火灾危险场景下的短期风险缓解以及水上的警报->备用操作->操作员接管。10秒以下的模型保留了较慢最先进模型的大部分感知能力。备用操作选择器优于仅基于几何的基线方法，并在火灾场景中增加了 standoff 距离。现场运行验证了端到端操作。这些结果支持视觉-语言模型作为符合国际海事组织MASS法规草案的语义备用操作选择器，在实际延迟预算内运行，并激励未来在领域自适应混合自主系统方面的研究，该系统将基础模型语义与多传感器鸟瞰感知和短期重新规划配对。


### 论文摘要

The draft IMO MASS Code requires autonomous and remotely supervised maritime vessels to detect departures from their operational design domain, enter a predefined fallback that notifies the operator, permit immediate human override, and avoid changing the voyage plan without approval. Meeting these obligations in the alert-to-takeover gap calls for a short-horizon, human-overridable fallback maneuver. Classical maritime autonomy stacks struggle when the correct action depends on meaning (e.g., diver-down flag means people in the water, fire close by means hazard). We argue (i) that vision-language models (VLMs) provide semantic awareness for such out-of-distribution situations, and (ii) that a fast-slow anomaly pipeline with a short-horizon, human-overridable fallback maneuver makes this practical in the handover window. We introduce Semantic Lookout, a camera-only, candidate-constrained vision-language model (VLM) fallback maneuver selector that selects one cautious action (or station-keeping) from water-valid, world-anchored trajectories under continuous human authority. On 40 harbor scenes we measure per-call scene understanding and latency, alignment with human consensus (model majority-of-three voting), short-horizon risk-relief on fire hazard scenes, and an on-water alert->fallback maneuver->operator handover. Sub-10 s models retain most of the awareness of slower state-of-the-art models. The fallback maneuver selector outperforms geometry-only baselines and increases standoff distance on fire scenes. A field run verifies end-to-end operation. These results support VLMs as semantic fallback maneuver selectors compatible with the draft IMO MASS Code, within practical latency budgets, and motivate future work on domain-adapted, hybrid autonomy that pairs foundation-model semantics with multi-sensor bird's-eye-view perception and short-horizon replanning.

---

## 38. Spatial-aware Vision Language Model for Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2512.24331v1](http://arxiv.org/abs/2512.24331v1)

**作者:** Weijie Wei, Zhipeng Luo, Ling Feng, Venice Erin Liong

**发布时间:** 2025-12-30

### GPT解析

### 总结

这篇论文提出了一个名为LVLDrive的新型框架，用于升级现有的视觉语言模型(VLMs)，通过整合激光雷达点云作为额外输入模态，为自动驾驶提供强大的3D度量空间理解能力。

### 背景

视觉语言模型(VLMs)利用语言模型中嵌入的常识，在端到端自动驾驶方面显示出巨大潜力，但它们对2D图像线索的依赖成为安全性和可靠性的关键瓶颈。当前基于图像的方法在准确的度量空间推理和几何推断方面存在困难，导致不可靠的驾驶策略。

### 目的

为了解决现有VLMs在自动驾驶场景中的局限性，提出LVLDrive框架，为现有VLMs提供强大的3D度量空间理解能力，以提高自动驾驶的安全性和可靠性。

### 方法

提出了LVLDrive(LiDAR-Vision-Language)框架，通过整合激光雷达点云作为额外输入模态来升级现有的VLMs。引入了渐进式融合Q-Former，逐步注入激光雷达特征，确保VLMs现有知识库的稳定性和保留。同时，开发了一个空间感知问答(SA-QA)数据集，明确教授模型高级3D感知和推理能力。

### 主要发现

在驾驶基准测试上的广泛实验表明，LVLDrive在场景理解、度量空间感知和可靠的驾驶决策方面，相比仅使用视觉的同类方法实现了优越的性能。

### 结论

研究强调了明确的3D度量数据对于构建可信的基于VLMs的自主系统的必要性。

### 翻译

视觉语言模型(VLMs)通过利用语言模型中嵌入的常识，在端到端自动驾驶方面显示出巨大潜力，但它们对2D图像线索的依赖成为安全性和可靠性的关键瓶颈。当前基于图像的方法在准确的度量空间推理和几何推断方面存在困难，导致不可靠的驾驶策略。为了弥补这一差距，我们提出了LVLDrive(LiDAR-Vision-Language)，这是一个新型框架，专门设计用于通过整合激光雷达点云作为额外输入模态，升级现有的VLMs，为自动驾驶提供强大的3D度量空间理解能力。一个关键挑战在于减轻不兼容的3D数据对预训练VLMs带来的灾难性干扰。为此，我们引入了一个渐进式融合Q-Former，逐步注入激光雷达特征，确保VLMs现有知识库的稳定性和保留。此外，我们开发了一个空间感知问答(SA-QA)数据集，明确教授模型高级3D感知和推理能力。在驾驶基准测试上的广泛实验表明，LVLDrive在场景理解、度量空间感知和可靠的驾驶决策方面，相比仅使用视觉的同类方法实现了优越的性能。我们的工作强调了明确的3D度量数据对于构建可信的基于VLMs的自主系统的必要性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有视觉语言模型(VLMs)在自动驾驶应用中的关键瓶颈：过度依赖2D图像进行场景理解和决策-making，导致在精确的3D空间推理和几何推理方面表现不足。这个问题在现实中至关重要，因为自动驾驶需要准确判断距离、范围、遮挡和物体交互等3D信息，这些是安全规划的先决条件，特别是在复杂的城市环境中。仅从2D图像推断几何信息在遮挡、恶劣天气等情况下不可靠，会影响自动驾驶系统的安全性和可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者的思考过程始于识别VLMs在3D空间理解方面的局限性，然后考虑引入LiDAR点云作为补充信息。他们发现主要挑战在于如何将未对齐的3D LiDAR数据注入已优化在图像-文本数据上的VLMs而不破坏其已有知识。为此，作者设计了渐进式融合Q-Former结构，通过门控注意力机制逐步引入LiDAR特征。该方法借鉴了现有工作：基于Q-Former架构进行改进，参考OmniDrive的Q-Former 3D块设计，从LLaMA-Adapter学习零初始化门控机制，并基于nuScenes数据集构建SA-QA数据集。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过渐进式融合机制将LiDAR提供的3D度量空间信息与预训练VLM结合，同时保持模型已有的视觉-语言知识库稳定性。实现流程包括：1)接收文本、图像和点云三种输入；2)使用三个预训练编码器分别处理不同模态数据；3)通过渐进式融合Q-Former融合多模态特征，使用零初始化门控机制逐步引入点云特征；4)将融合特征输入大型语言模型生成任务特定响应；5)使用SA-QA数据集进行微调增强空间理解能力。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)渐进式融合Q-Former结构，通过零初始化门控机制逐步注入LiDAR特征；2)空间感知问答数据集(SA-QA)，包含空间感知和推理任务，强制模型进行跨模态推理；3)统一的多模态融合策略，同时处理图像和点云特征。相比之前工作，LVLDrive不同于纯图像VLMs(缺乏3D理解)、简单多模态融合(未解决特征对齐)和专用3D模型(缺乏语言能力)，通过创新融合机制和数据集，实现了3D感知与语言推理的有效结合。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LVLDrive通过创新的渐进式融合机制和空间感知数据集，成功地将LiDAR的3D度量空间理解能力与视觉语言模型的强大推理能力相结合，显著提升了自动驾驶系统在场景理解、空间感知和可靠决策方面的性能。'}


### 论文摘要

While Vision-Language Models (VLMs) show significant promise for end-to-end autonomous driving by leveraging the common sense embedded in language models, their reliance on 2D image cues for complex scene understanding and decision-making presents a critical bottleneck for safety and reliability. Current image-based methods struggle with accurate metric spatial reasoning and geometric inference, leading to unreliable driving policies. To bridge this gap, we propose LVLDrive (LiDAR-Vision-Language), a novel framework specifically designed to upgrade existing VLMs with robust 3D metric spatial understanding for autonomous driving by incoperating LiDAR point cloud as an extra input modality. A key challenge lies in mitigating the catastrophic disturbance introduced by disparate 3D data to the pre-trained VLMs. To this end, we introduce a Gradual Fusion Q-Former that incrementally injects LiDAR features, ensuring the stability and preservation of the VLM's existing knowledge base. Furthermore, we develop a spatial-aware question-answering (SA-QA) dataset to explicitly teach the model advanced 3D perception and reasoning capabilities. Extensive experiments on driving benchmarks demonstrate that LVLDrive achieves superior performance compared to vision-only counterparts across scene understanding, metric spatial perception, and reliable driving decision-making. Our work highlights the necessity of explicit 3D metric data for building trustworthy VLM-based autonomous systems.

---

## 39. GeoBench: Rethinking Multimodal Geometric Problem-Solving via Hierarchical Evaluation

**论文链接:** [http://arxiv.org/abs/2512.24119v1](http://arxiv.org/abs/2512.24119v1)

**作者:** Yuan Feng, Yue Yang, Xiaohan He, Jiatong Zhao, Jianlong Chen, Zijun Chen, Daocheng Fu, Qi Liu, Renqiu Xia, Bo Zhang, Junchi Yan

**发布时间:** 2025-12-30

### GPT解析

### 总结

该研究提出了GeoBench，一个用于评估视觉语言模型几何问题解决能力的分层基准，包含四个推理层级，并通过六个形式验证的任务进行了系统评估。

### 背景

几何问题解决是数学推理的关键分支，需要精确分析形状和空间关系。当前对视觉语言模型几何推理能力的评估存在局限性，包括测试数据污染风险、过分强调最终答案而非推理过程，以及诊断粒度不足。

### 目的

解决现有几何推理评估方法的局限性，提供一个全面的基准来系统评估几何问题解决能力，并为相关系统开发提供指导。

### 方法

提出GeoBench基准，包含四个几何问题解决推理层级：视觉感知、目标导向规划、严格定理应用和自我反思回溯。通过TrustGeoGen生成的六个经过形式验证的任务，系统评估从属性提取到逻辑错误纠正的各种能力。

### 主要发现

推理模型(如OpenAI-o3)优于通用多语言大模型，但性能随任务复杂度增加而显著下降；子目标分解和无关前提过滤对问题解决准确性有关键影响；思维链提示在某些任务中降低了性能。

### 结论

GeoBench作为一个全面的几何问题解决评估基准被建立，同时为开发几何问题解决系统提供了可行的指导方针。

### 翻译

几何问题解决构成了数学推理的关键分支，需要对形状和空间关系进行精确分析。当前对视觉语言模型中几何推理能力的评估存在局限性，包括基于教科书基准的测试数据污染风险、过分强调最终答案而非推理过程，以及诊断粒度不足。为解决这些问题，我们提出了GeoBench，一个包含几何问题解决四个推理层级的分层基准：视觉感知、目标导向规划、严格定理应用和自我反思回溯。通过TrustGeoGen生成的六个经过形式验证的任务，我们系统评估了从属性提取到逻辑错误纠正的各种能力。实验表明，虽然像OpenAI-o3这样的推理模型优于通用多语言大模型，但随着任务复杂度的增加，性能显著下降。主要发现表明，子目标分解和无关前提过滤对最终问题解决准确性有关键影响，而思维链提示在某些任务中意外降低了性能。这些发现确立了GeoBench作为全面基准的地位，同时为开发几何问题解决系统提供了可行的指导方针。


### 论文摘要

Geometric problem solving constitutes a critical branch of mathematical reasoning, requiring precise analysis of shapes and spatial relationships. Current evaluations of geometric reasoning in vision-language models (VLMs) face limitations, including the risk of test data contamination from textbook-based benchmarks, overemphasis on final answers over reasoning processes, and insufficient diagnostic granularity. To address these issues, we present GeoBench, a hierarchical benchmark featuring four reasoning levels in geometric problem-solving: Visual Perception, Goal-Oriented Planning, Rigorous Theorem Application, and Self-Reflective Backtracking. Through six formally verified tasks generated via TrustGeoGen, we systematically assess capabilities ranging from attribute extraction to logical error correction. Experiments reveal that while reasoning models like OpenAI-o3 outperform general MLLMs, performance declines significantly with increasing task complexity. Key findings demonstrate that sub-goal decomposition and irrelevant premise filtering critically influence final problem-solving accuracy, whereas Chain-of-Thought prompting unexpectedly degrades performance in some tasks. These findings establish GeoBench as a comprehensive benchmark while offering actionable guidelines for developing geometric problem-solving systems.

---

## 40. ColaVLA: Leveraging Cognitive Latent Reasoning for Hierarchical Parallel Trajectory Planning in Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2512.22939v2](http://arxiv.org/abs/2512.22939v2)

**作者:** Qihang Peng, Xuesong Chen, Chenye Yang, Shaoshuai Shi, Hongsheng Li

**发布时间:** 2025-12-28

**备注:** 11 pages, 4 figures. Project page: https://pqh22.github.io/projects/ColaVLA/index.html

### GPT解析

### 总结

ColaVLA是一个统一的视觉-语言-动作框架，通过认知潜在推理器和分层并行规划器解决了当前基于视觉语言模型的自动驾驶规划者面临的三个关键挑战，实现了高效、准确和安全的轨迹生成。

### 背景

自动驾驶需要从复杂的多模态输入生成安全可靠的轨迹。传统方法将感知、预测和规划分离，而端到端系统联合学习这些功能。视觉语言模型通过引入跨模态先验和常识推理丰富了这一范式。

### 目的

解决当前基于视觉语言模型的自动驾驶规划者面临的三个关键挑战：(i)离散文本推理与连续控制之间的不匹配；(ii)自回归思维链解码导致的高延迟；(iii)低效或非因果的规划者限制了实时部署。

### 方法

提出ColaVLA框架，包含认知潜在推理器和分层并行规划器。认知潜在推理器通过自我自适应选择将场景理解压缩为紧凑的、面向决策的元动作嵌入，只需两次VLM前向传播。分层并行规划器在单次前向传播中生成多尺度、因果一致的轨迹。

### 主要发现

在nuScenes基准测试中，ColaVLA在开环和闭环设置中都取得了最先进的性能，具有有利的效率和鲁棒性。该框架保留了视觉语言模型的泛化性和可解释性，同时实现高效、准确和安全的轨迹生成。

### 结论

ColaVLA成功解决了当前VLM规划者的三个关键挑战，实现了高效、准确和安全的自动驾驶轨迹生成，为自动驾驶领域提供了一个有前景的解决方案。

### 翻译

自动驾驶需要从复杂的多模态输入生成安全可靠的轨迹。传统的模块化流程将感知、预测和规划分开，而最近的端到端系统则联合学习它们。视觉语言模型通过引入跨模态先验和常识推理进一步丰富了这一范式，然而，当前基于VLM的规划者面临三个关键挑战：(i)离散文本推理与连续控制之间的不匹配，(ii)自回归思维链解码导致的高延迟，以及(iii)低效或非因果的规划者限制了实时部署。我们提出了ColaVLA，一个统一的视觉-语言-动作框架，它将推理从文本转移到统一的潜在空间，并与分层并行轨迹解码器耦合。认知潜在推理器通过自我自适应选择将场景理解压缩为紧凑的、面向决策的元动作嵌入，只需两次VLM前向传播。然后，分层并行规划器在单次前向传播中生成多尺度、因果一致的轨迹。这些组件共同保留了VLM的泛化性和可解释性，同时实现了高效、准确和安全的轨迹生成。在nuScenes基准测试中，ColaVLA在开环和闭环设置中都取得了最先进的性能，具有有利的效率和鲁棒性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决基于视觉-语言模型(VLM)的自动驾驶规划器面临的三个关键挑战：离散文本推理与连续控制之间的不匹配、自回归链式思维推理带来的高延迟、以及效率低下或非因果的规划器限制实时部署。这个问题在现实中非常重要，因为自动驾驶系统需要从复杂多模态输入中生成安全可靠的轨迹，而这些挑战限制了VLM在实际部署中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先回顾了自动驾驶系统从模块化流水线到端到端方法再到VLM集成的发展历程，指出现有VLM规划器的三个关键挑战。他们受大型语言模型在潜在空间推理方面进展的启发，决定将推理从显式文本思维链转移到统一潜在空间，并与保持因果关系的并行解码规划器配对。该方法借鉴了现有视觉-语言模型、端到端自动驾驶方法和潜在空间推理技术，通过自我适应选择和元信息压缩设计认知潜在推理器，并利用多尺度解码设计层次并行规划器。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将推理从文本转移到统一的潜在空间，并与层次并行规划器耦合，避免模态不匹配，减少推理延迟，保持因果一致性。整体流程分为两大部分：1)认知潜在推理：包括场景理解(构建多模态输入序列并通过VLM处理)、关键实体识别(通过自我适应路由器选择安全相关视觉标记)、潜在思考(连接选定标记和元查询进行第二前向传播)、战略决策合成(通过FiLM调制和MLP生成决策)；2)层次并行规划：包括阶段感知轨迹查询(将预测时间划分为多尺度)、因果保持混合注意力(设计特殊注意力掩码)、置信度引导并行解码(同时处理多个候选策略)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的视觉-语言-动作框架，直接在连续轨迹上操作；2)认知潜在推理器，将推理转移到潜在空间；3)层次并行规划器，在单次前向传播中解码多尺度轨迹；4)因果保持混合注意力机制。相比之前的工作，ColaVLA避免了文本方法的自回归推理高延迟问题，克服了传统端到端方法缺乏高级认知指导的局限，解决了模块化系统的接口脆弱性问题，并将VLM前向传播次数减少了5倍以上，同时保持了高精度和安全性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ColaVLA通过将视觉-语言模型的推理从文本转移到统一的潜在空间，并配合层次并行规划器，实现了高效、准确且安全的自动驾驶轨迹规划，在保持模型可解释性的同时显著降低了推理延迟。'}


### 论文摘要

Autonomous driving requires generating safe and reliable trajectories from complex multimodal inputs. Traditional modular pipelines separate perception, prediction, and planning, while recent end-to-end (E2E) systems learn them jointly. Vision-language models (VLMs) further enrich this paradigm by introducing cross-modal priors and commonsense reasoning, yet current VLM-based planners face three key challenges: (i) a mismatch between discrete text reasoning and continuous control, (ii) high latency from autoregressive chain-of-thought decoding, and (iii) inefficient or non-causal planners that limit real-time deployment. We propose ColaVLA, a unified vision-language-action framework that transfers reasoning from text to a unified latent space and couples it with a hierarchical, parallel trajectory decoder. The Cognitive Latent Reasoner compresses scene understanding into compact, decision-oriented meta-action embeddings through ego-adaptive selection and only two VLM forward passes. The Hierarchical Parallel Planner then generates multi-scale, causality-consistent trajectories in a single forward pass. Together, these components preserve the generalization and interpretability of VLMs while enabling efficient, accurate and safe trajectory generation. Experiments on the nuScenes benchmark show that ColaVLA achieves state-of-the-art performance in both open-loop and closed-loop settings with favorable efficiency and robustness.

---

## 41. Break Out the Silverware -- Semantic Understanding of Stored Household Items

**论文链接:** [http://arxiv.org/abs/2512.23739v1](http://arxiv.org/abs/2512.23739v1)

**作者:** Michaela Levi-Richter, Reuth Mirsky, Oren Glickman

**发布时间:** 2025-12-25

**备注:** Poster presented at the Israeli Seminar on Computational Linguistics 2025

### GPT解析

### 总结

该论文提出了家庭服务机器人面临的物品存储位置推断挑战，引入了'存储家庭物品挑战'作为评估服务机器人认知能力的基准任务，并开发了NOAM模型来解决这一难题。NOAM结合视觉理解和语言模型推理，显著提高了预测物品存储位置的准确性，接近人类水平。

### 背景

尽管视觉和操作技术取得了进步，家庭服务机器人仍缺乏推断日常物品存储位置所需的常识推理能力。物品通常被放置在看不见的抽屉、橱柜或壁橱中，给机器人完成任务带来挑战。

### 目的

引入'存储家庭物品挑战'作为评估服务机器人认知能力的基准任务，并开发能够预测家庭物品最可能存储位置的模型。

### 方法

提出NOAM（不可见物体分配模型），一种结合结构化场景理解和大型语言模型推理的混合智能体管道。NOAM将视觉输入转换为空间上下文和可见容器的自然语言描述，然后提示语言模型（如GPT-4）推断最可能的隐藏存储位置。

### 主要发现

NOAM在预测准确性方面显著优于多种基线方法，包括随机选择、视觉语言管道和领先的多模态模型，表现接近人类水平，证明了其在家庭物品位置推断任务中的有效性。

### 结论

NOAM集成的视觉-语言智能体表现出涌现的常识推理能力，设计用于在更广泛的机器人系统中进行模块化部署，突显了在家庭环境中部署具有认知能力的智能体的最佳实践。

### 翻译

'给我一个盘子。'对于家庭服务机器人来说，这个简单的命令揭示了一个复杂的挑战：推断日常物品的存储位置，这些物品通常被放在看不见的抽屉、橱柜或壁橱中。尽管视觉和操作技术取得了进步，机器人仍然缺乏完成此任务所需的常识推理能力。我们引入了存储家庭物品挑战，这是一个用于评估服务机器人认知能力的基准任务：给定一个家庭场景和一个查询物品，预测其最可能的存储位置。我们的基准包括两个数据集：(1)一个来自参与者厨房的100个物品-图像对的真实世界评估集，具有人工注释的真实标签，以及(2)一个在公开厨房图像上注释了存储多边形的6,500个物品-图像对开发集。这些数据集支持对家庭组织的现实建模，并能够跨智能体架构进行比较评估。为了开始应对这一挑战，我们引入了NOAM（不可见物体分配模型），这是一种结合结构化场景理解和大型语言模型推理的混合智能体管道。NOAM将视觉输入转换为空间上下文和可见容器的自然语言描述，然后提示语言模型（如GPT-4）推断最可能的隐藏存储位置。这种集成的视觉-语言智能体表现出涌现的常识推理能力，设计用于在更广泛的机器人系统中进行模块化部署。我们针对包括随机选择、视觉-语言管道（Grounding-DINO + SAM）、领先的多模态模型（如Gemini、GPT-4o、Kosmos-2、LLaMA、Qwen）和人类性能在内的基线评估了NOAM。NOAM显著提高了预测准确性，并接近人类水平的结果，突显了在家庭环境中部署具有认知能力的智能体的最佳实践。


### 论文摘要

``Bring me a plate.'' For domestic service robots, this simple command reveals a complex challenge: inferring where everyday items are stored, often out of sight in drawers, cabinets, or closets. Despite advances in vision and manipulation, robots still lack the commonsense reasoning needed to complete this task. We introduce the Stored Household Item Challenge, a benchmark task for evaluating service robots' cognitive capabilities: given a household scene and a queried item, predict its most likely storage location.   Our benchmark includes two datasets: (1) a real-world evaluation set of 100 item-image pairs with human-annotated ground truth from participants' kitchens, and (2) a development set of 6,500 item-image pairs annotated with storage polygons over public kitchen images. These datasets support realistic modeling of household organization and enable comparative evaluation across agent architectures.   To begin tackling this challenge, we introduce NOAM (Non-visible Object Allocation Model), a hybrid agent pipeline that combines structured scene understanding with large language model inference. NOAM converts visual input into natural language descriptions of spatial context and visible containers, then prompts a language model (e.g., GPT-4) to infer the most likely hidden storage location. This integrated vision-language agent exhibits emergent commonsense reasoning and is designed for modular deployment within broader robotic systems.   We evaluate NOAM against baselines including random selection, vision-language pipelines (Grounding-DINO + SAM), leading multimodal models (e.g., Gemini, GPT-4o, Kosmos-2, LLaMA, Qwen), and human performance. NOAM significantly improves prediction accuracy and approaches human-level results, highlighting best practices for deploying cognitively capable agents in domestic environments.

---

## 42. From Static to Dynamic: Evaluating the Perceptual Impact of Dynamic Elements in Urban Scenes Using Generative Inpainting

**论文链接:** [http://arxiv.org/abs/2512.24513v1](http://arxiv.org/abs/2512.24513v1)

**作者:** Zhiwei Wei, Mengzi Zhang, Boyan Lu, Zhitao Deng, Nai Yang, Hua Liao

**发布时间:** 2025-12-30

**备注:** 31 pages, 5 figures

### GPT解析

### 总结

该研究探讨了街景图像中动态元素对城市感知的影响，通过构建配对图像进行实验，发现移除行人和车辆等动态元素会导致感知活力显著下降30.97%，而其他感知维度变化较小。研究还确定了照明条件、人类存在和深度变化是影响感知的关键因素，并指出基于静态图像的城市评估可能低估城市活力。

### 背景

从街景图像理解城市感知已成为城市分析和以人为本的城市设计的中心话题，但现有研究大多将城市场景视为静态的，忽略了行人和车辆等动态元素的作用，这可能导致基于感知的城市分析存在潜在偏见。

### 目的

解决现有研究中的偏见问题，提出一个控制框架，通过语义分割和MLLM引导的生成修复技术，构建包含和不包含行人和车辆的配对街景图像，以分离动态元素的感知效果。

### 方法

基于中国东莞720张配对图像，进行感知实验，参与者评估原始和编辑场景在六个感知维度上的表现。然后使用多模态视觉特征训练11个机器学习模型，探索潜在机制。最后将训练好的模型扩展到城市规模数据集，预测移除动态元素后的活力变化。

### 主要发现

移除动态元素导致感知到的活力一致下降30.97%，其他维度的变化较为温和且异质；照明条件、人类存在和深度变化是驱动感知变化的关键因素；在个体层面，65%的参与者表现出显著的活力变化，而其他维度为35-50%；性别对安全感知有轻微的调节作用；城市级别结果显示，这种感知变化广泛且具有空间结构性，影响73.7%的位置和32.1%的图像。

### 结论

仅基于静态图像的城市感知评估可能会大大低估城市的活力，城市规划和设计应考虑动态元素对感知的重要影响。

### 翻译

从街景图像理解城市感知已成为城市分析和以人为本的城市设计的中心话题。然而，大多数现有研究将城市场景视为静态的，很大程度上忽略了行人和车辆等动态元素的作用，引发了基于感知的城市分析可能存在潜在偏见的担忧。为解决这个问题，我们提出了一个控制框架，通过语义分割和MLLM引导的生成修复技术，构建包含和不包含行人和车辆的配对街景图像，以分离动态元素的感知效果。基于中国东莞的720张配对图像，进行了一项感知实验，参与者在六个感知维度上评估原始和编辑场景。结果表明，移除动态元素导致感知到的活力一致下降30.97%，而其他维度的变化较为温和且异质。为进一步探索潜在机制，我们使用多模态视觉特征训练了11个机器学习模型，并确定照明条件、人类存在和深度变化是驱动感知变化的关键因素。在个体层面，65%的参与者表现出显著的活力变化，而其他维度为35-50%；性别进一步显示出对安全感知的轻微调节作用。除了受控实验外，训练好的模型还被扩展到城市规模数据集，以预测移除动态元素后的活力变化。城市级别结果显示，这种感知变化是广泛且具有空间结构的，影响73.7%的位置和32.1%的图像，这表明仅基于静态图像的城市感知评估可能会大大低估城市的活力。


### 论文摘要

Understanding urban perception from street view imagery has become a central topic in urban analytics and human centered urban design. However, most existing studies treat urban scenes as static and largely ignore the role of dynamic elements such as pedestrians and vehicles, raising concerns about potential bias in perception based urban analysis. To address this issue, we propose a controlled framework that isolates the perceptual effects of dynamic elements by constructing paired street view images with and without pedestrians and vehicles using semantic segmentation and MLLM guided generative inpainting. Based on 720 paired images from Dongguan, China, a perception experiment was conducted in which participants evaluated original and edited scenes across six perceptual dimensions. The results indicate that removing dynamic elements leads to a consistent 30.97% decrease in perceived vibrancy, whereas changes in other dimensions are more moderate and heterogeneous. To further explore the underlying mechanisms, we trained 11 machine learning models using multimodal visual features and identified that lighting conditions, human presence, and depth variation were key factors driving perceptual change. At the individual level, 65% of participants exhibited significant vibrancy changes, compared with 35-50% for other dimensions; gender further showed a marginal moderating effect on safety perception. Beyond controlled experiments, the trained model was extended to a city-scale dataset to predict vibrancy changes after the removal of dynamic elements. The city level results reveal that such perceptual changes are widespread and spatially structured, affecting 73.7% of locations and 32.1% of images, suggesting that urban perception assessments based solely on static imagery may substantially underestimate urban liveliness.

---

## 43. A multimodal Transformer for InSAR-based ground deformation forecasting with cross-site generalization across Europe

**论文链接:** [http://arxiv.org/abs/2512.23906v1](http://arxiv.org/abs/2512.23906v1)

**作者:** Wendong Yao, Binhua Huang, Soumyabrata Dev

**发布时间:** 2025-12-30

**备注:** submitted to ISPRS Journal of Photogrammetry and Remote Sensing for review

### GPT解析

### 总结

该研究提出了一种多模态基于块的Transformer模型，用于从EGMS时间序列中预测下一时间段的地面位移图，在测试中表现出色，为近实时区域尺度地面变形监测提供了有效工具。

### 背景

近实时区域尺度地面变形监测对城市规划、关键基础设施管理和自然灾害减灾日益重要。虽然InSAR和EGMS等服务提供了密集的过去运动观测，但由于长期趋势、季节性周期、突然不连续性和空间异质性的叠加，预测下一个观测仍具挑战性。

### 目的

开发一种能够进行单步、固定间隔的下一时段位移图预测的模型，解决地面变形预测中的挑战。

### 方法

研究提出多模态基于块的Transformer模型，从EGMS时间序列（重新采样为100公里×100公里瓦片上的64×64网格）预测位移图。模型接收最近位移快照、静态运动指标（平均速度、加速度、季节振幅）和年编码。在爱尔兰东部瓦片(E32N34)上进行了测试。

### 主要发现

在仅位移数据设置中，STGCN表现最强；但当所有模型接收相同多模态输入时，多模态Transformer明显优于CNN-LSTM、CNN-LSTM+Attn和多模态STGCN，实现RMSE=0.90毫米和R²=0.97的最佳测试结果。

### 结论

多模态Transformer模型能有效整合多种数据源进行准确的下一时段位移预测，为近实时区域尺度地面变形监测提供了有效解决方案。

### 翻译

近实时区域尺度地面变形监测日益需要支持城市规划、关键基础设施管理和自然灾害减灾。虽然干涉合成孔径雷达(InSAR)和欧洲地面运动服务(EGMS)等大陆级服务提供了密集的过去运动观测，但由于长期趋势、季节性周期和偶尔的突然不连续性（如地震阶跃）以及强空间异质性的叠加，预测下一个观测仍然具有挑战性。在本研究中，我们提出了一种多模态基于块的Transformer，用于从EGMS时间序列（重新采样为100公里×100公里瓦片上的64×64网格）进行单步、固定间隔的下一时段位移图预测。该模型接收最近的位移快照以及(i)仅从训练窗口以泄漏安全方式计算的静态运动指标（平均速度、加速度、季节振幅），和(ii)谐波日-of-year编码。在爱尔兰东部瓦片(E32N34)上，在仅使用位移数据的设置中，STGCN表现最强；但当所有模型接收相同的多模态输入时，多模态Transformer明显优于CNN-LSTM、CNN-LSTM+Attn和多模态STGCN，在测试集上实现了RMSE = 0.90毫米和R² = 0.97的最佳阈值准确率。


### 论文摘要

Near-real-time regional-scale monitoring of ground deformation is increasingly required to support urban planning, critical infrastructure management, and natural hazard mitigation. While Interferometric Synthetic Aperture Radar (InSAR) and continental-scale services such as the European Ground Motion Service (EGMS) provide dense observations of past motion, predicting the next observation remains challenging due to the superposition of long-term trends, seasonal cycles, and occasional abrupt discontinuities (e.g., co-seismic steps), together with strong spatial heterogeneity. In this study we propose a multimodal patch-based Transformer for single-step, fixed-interval next-epoch nowcasting of displacement maps from EGMS time series (resampled to a 64x64 grid over 100 km x 100 km tiles). The model ingests recent displacement snapshots together with (i) static kinematic indicators (mean velocity, acceleration, seasonal amplitude) computed in a leakage-safe manner from the training window only, and (ii) harmonic day-of-year encodings. On the eastern Ireland tile (E32N34), the STGCN is strongest in the displacement-only setting, whereas the multimodal Transformer clearly outperforms CNN-LSTM, CNN-LSTM+Attn, and multimodal STGCN when all models receive the same multimodal inputs, achieving RMSE = 0.90 mm and $R^2$ = 0.97 on the test set with the best threshold accuracies.

---

## 44. Subsecond 3D Mesh Generation for Robot Manipulation

**论文链接:** [http://arxiv.org/abs/2512.24428v1](http://arxiv.org/abs/2512.24428v1)

**作者:** Qian Wang, Omar Abdellall, Tony Gao, Xiatao Sun, Daniel Rakita

**发布时间:** 2025-12-30

**备注:** In submission

### GPT解析

### 总结

该研究介绍了一个端到端系统，能够在不到一秒的时间内从单个RGB-D图像生成高质量、上下文感知的3D网格，解决了3D网格生成速度慢和上下文感知不足的问题。

### 背景

3D网格是计算机科学和工程学中的基本表示形式，在机器人学中尤为重要，因为它们以与机器人物理世界交互直接对应的形式捕捉物体，实现稳定抓取预测、碰撞检测和动力学模拟等核心功能。

### 目的

解决两个关键挑战：1) 高保真度网格生成速度过慢，无法满足实时使用需求；2) 网格生成本身不足，需要正确分割场景并具有适当的尺度和姿态。

### 方法

开发了一个端到端系统，集成了开放词汇目标分割、加速的基于扩散的网格生成和鲁棒点云配准，每个组件都针对速度和准确性进行了优化。

### 主要发现

该系统能够在不到一秒的时间内从单个RGB-D图像生成高质量、上下文感知的3D网格，并在实际操作任务中展示了其有效性。

### 结论

该系统使网格能够成为机器人感知和规划的实用、按需表示方法，解决了实时机器人感知的瓶颈问题。

### 翻译

3D网格是计算机科学和工程学中广泛使用的基本表示形式。在机器人学中，它们特别有价值，因为它们以与机器人物理世界交互直接对应的形式捕捉物体，实现了稳定抓取预测、碰撞检测和动力学模拟等核心功能。尽管近年来自动3D网格生成方法已显示出有希望的进展，可能为实时机器人感知提供了一条途径，但两个关键挑战仍然存在。首先，生成高保真度网格对于实时使用来说慢得令人望而却步，通常每个物体需要几十秒。其次，网格生成本身是不够的。在机器人学中，网格必须是上下文感知的，即从场景中正确分割出来，并具有适当的尺度和姿态。此外，除非这些上下文感知步骤保持高效，否则它们只会引入新的瓶颈。在这项工作中，我们介绍了一个端到端系统来解决这些挑战，能够在不到一秒的时间内从单个RGB-D图像生成高质量、上下文感知的3D网格。我们的流程集成了开放词汇目标分割、加速的基于扩散的网格生成和鲁棒点云配准，每个组件都针对速度和准确性进行了优化。我们在实际操作任务中展示了其有效性，表明它使网格能够成为机器人感知和规划的实用、按需表示方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决两个关键问题：一是高保真3D网格生成速度太慢，无法满足实时机器人应用需求；二是网格生成后还需要正确地从场景中分割出来，并确定正确的尺度和姿态。这些问题在机器人领域很重要，因为3D网格能直接捕捉物体形式，与机器人如何与物理世界交互的方式相一致，对预测稳定抓取、检测碰撞和模拟动态等核心能力至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到高保真3D网格在机器人中的价值，但现有方法速度太慢；同时发现即使生成网格，还需要正确分割和定位才能用于机器人。他们设计了一个端到端系统，结合开放词汇分割、加速网格生成和鲁棒配准。该方法借鉴了多项现有工作：使用Hunyuan3D 2.0作为网格生成基础但通过FlashVDM加速；使用SAM2进行图像分割；使用Florence-2进行开放词汇检测；使用Depth Anything v2进行深度估计；使用RANSAC和ICP进行点云配准。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将高质量3D网格生成与开放词汇分割和鲁棒点云配准相结合，创建端到端系统，并通过渐进式流蒸馏和分层SDF解码等技术加速高质量3D生成模型，专注于几何而非纹理。整体流程分三阶段：1)开放词汇分割(0.2秒)：使用Florence-2检测和SAM2细化掩码，用Depth Anything v2增强深度；2)加速网格生成(0.5秒)：使用FlashVDM蒸馏的Hunyuan3D 2.0，采用分层SDF解码；3)物体配准(0.15秒)：使用RANSAC和ICP将网格与观测点云对齐。总运行时间0.85秒。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首个能在不到一秒内生成高质量、上下文感知3D网格的端到端系统；2)通过FlashVDM将扩散步骤从几十步减少到仅3步；3)采用分层SDF解码降低体积解码成本90%以上；4)专注于几何而非纹理；5)结合开放词汇分割、加速网格生成和鲁棒点云配准。相比之前工作，本文打破了速度与质量之间的权衡（图2），将网格获取从慢速离线过程转变为实时工具，解决了之前方法要么速度快但质量低、要么质量高但速度慢的问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文开发了一个能在不到一秒内从RGB-D图像生成高质量、上下文感知3D网格的端到端系统，为实时机器人应用提供了实用的按需3D表示方法。'}


### 论文摘要

3D meshes are a fundamental representation widely used in computer science and engineering. In robotics, they are particularly valuable because they capture objects in a form that aligns directly with how robots interact with the physical world, enabling core capabilities such as predicting stable grasps, detecting collisions, and simulating dynamics. Although automatic 3D mesh generation methods have shown promising progress in recent years, potentially offering a path toward real-time robot perception, two critical challenges remain. First, generating high-fidelity meshes is prohibitively slow for real-time use, often requiring tens of seconds per object. Second, mesh generation by itself is insufficient. In robotics, a mesh must be contextually grounded, i.e., correctly segmented from the scene and registered with the proper scale and pose. Additionally, unless these contextual grounding steps remain efficient, they simply introduce new bottlenecks. In this work, we introduce an end-to-end system that addresses these challenges, producing a high-quality, contextually grounded 3D mesh from a single RGB-D image in under one second. Our pipeline integrates open-vocabulary object segmentation, accelerated diffusion-based mesh generation, and robust point cloud registration, each optimized for both speed and accuracy. We demonstrate its effectiveness in a real-world manipulation task, showing that it enables meshes to be used as a practical, on-demand representation for robotics perception and planning.

---

## 45. Early Prediction of Sepsis using Heart Rate Signals and Genetic Optimized LSTM Algorithm

**论文链接:** [http://arxiv.org/abs/2512.24253v1](http://arxiv.org/abs/2512.24253v1)

**作者:** Alireza Rafiei, Farshid Hajati, Alireza Rezaee, Amirhossien Panahi, Shahadat Uddin

**发布时间:** 2025-12-30

### GPT解析

### 总结

本研究引入并评估了四种新型机器学习算法，通过分析心率数据在可穿戴设备上预测脓毒症发作，并通过遗传算法优化模型架构，研究结果显示可穿戴技术有望在ICU和病房环境外实现脓毒症早期检测。

### 背景

脓毒症是由感染引起的免疫系统失调反应，导致显著的死亡率、发病率和医疗成本。尽管已有针对ICU患者的多种预测模型，但在非病房环境中早期检测脓毒症的方法仍存在明显差距。

### 目的

引入并评估四种新型机器学习算法，通过分析心率数据在可穿戴设备上预测脓毒症发作。

### 方法

使用遗传算法优化模型架构，优化目标包括性能、计算复杂性和内存需求。为每个模型提取性能指标评估其在可穿戴设备上的实施可行性。模型最初针对一小时预测窗口定制，随后通过迁移学习扩展到四小时。

### 主要发现

研究取得了令人鼓舞的结果，表明可穿戴设备预测脓毒症的可行性。

### 结论

可穿戴技术有可能促进ICU和病房环境之外的脓毒症早期检测。

### 翻译

脓毒症是一种由感染引起的免疫系统失调反应，导致显著的死亡率、发病率和医疗成本。及时预测脓毒症进展对于通过早期干预减少不良结果至关重要。尽管已经开发了针对ICU患者的多种模型，但在非病房环境中早期检测脓毒症的方法仍然存在明显差距。本研究引入并评估了四种新型机器学习算法，旨在通过分析心率数据来预测可穿戴设备上的脓毒症发作。这些模型的架构通过遗传算法进行了优化，针对性能、计算复杂性和内存需求进行了优化。随后提取了每个模型的性能指标，以评估其在能够准确监测心率的可穿戴设备上实施的可行性。这些模型最初针对一小时预测窗口进行定制，随后通过迁移学习扩展到四小时。这项研究的令人鼓舞的结果表明，可穿戴技术有可能促进ICU和病房环境之外的脓毒症早期检测。


### 论文摘要

Sepsis, characterized by a dysregulated immune response to infection, results in significant mortality, morbidity, and healthcare costs. The timely prediction of sepsis progression is crucial for reducing adverse outcomes through early intervention. Despite the development of numerous models for Intensive Care Unit (ICU) patients, there remains a notable gap in approaches for the early detection of sepsis in non-ward settings. This research introduces and evaluates four novel machine learning algorithms designed for predicting the onset of sepsis on wearable devices by analyzing heart rate data. The architecture of these models was refined through a genetic algorithm, optimizing for performance, computational complexity, and memory requirements. Performance metrics were subsequently extracted for each model to evaluate their feasibility for implementation on wearable devices capable of accurate heart rate monitoring. The models were initially tailored for a prediction window of one hour, later extended to four hours through transfer learning. The encouraging outcomes of this study suggest the potential for wearable technology to facilitate early sepsis detection outside ICU and ward environments.

---

## 46. Beamforming for Massive MIMO Aerial Communications: A Robust and Scalable DRL Approach

**论文链接:** [http://arxiv.org/abs/2512.23902v1](http://arxiv.org/abs/2512.23902v1)

**作者:** Hesam Khoshkbari, Georges Kaddoum, Omid Abbasi, Bassant Selim, Halim Yanikomeroglu

**发布时间:** 2025-12-29

**DOI:** 10.1109/TCOMM.2025.3626652

### GPT解析

### 总结

本文提出了一种针对大规模多输入多输出非地面网络中空中平台站星座的分布式波束成形框架，通过基于熵的多智能体深度强化学习方法，在信道状态信息不完美条件下最大化下行链路和速率。

### 背景

研究大规模多输入多输出非地面网络中的分布式波束成形问题，涉及空中平台站星座，面临局部信道状态信息不完美的挑战。

### 目的

在信道状态信息不完美的条件下，最大化下行链路和速率，同时确保系统的可扩展性和鲁棒性。

### 方法

提出基于熵的多智能体深度强化学习方法，每个非地面基站使用傅里叶神经算子独立计算波束成形向量，并整合基于共轭先验机制的迁移学习和低秩分解技术以提高可扩展性和鲁棒性。

### 主要发现

所提方法在平均和速率、对CSI不完美的鲁棒性、用户移动性和可扩展性方面优于多种基线方案；与基于CNN和WMMSE的方法相比计算效率更高；与共享批评者DRL方法相比通信开销更小。

### 结论

该方法能有效支持非地面网络中的大规模部署，在性能和效率方面具有显著优势。

### 翻译

本文提出了一种针对大规模多输入多输出非地面网络中空中平台站星座的分布式波束成形框架，该框架在局部信道状态信息不完美的条件下旨在最大化下行链路和速率。我们提出了一种新颖的基于熵的多智能体深度强化学习方法，其中每个非地面基站使用傅里叶神经算子独立计算其波束成形向量，以捕获频域中的长程依赖关系。为确保可扩展性和鲁棒性，所提出的框架整合了基于共轭先验机制的迁移学习和低秩分解技术，从而能够有效支持大规模用户部署和航空层。我们的模拟结果表明，在平均和速率、对CSI不完美的鲁棒性、用户移动性以及跨越不同网络规模和用户密度的可扩展性方面，所提出的方法优于包括WMMSE、ZF、MRT、基于CNN的DRL和深度确定性策略梯度方法在内的基线方案。此外，我们表明与基于CNN和WMMSE的方法相比，所提出的方法实现了显著的计算效率，同时与共享批评者DRL方法相比减少了通信开销。


### 论文摘要

This paper presents a distributed beamforming framework for a constellation of airborne platform stations (APSs) in a massive Multiple-Input and Multiple-Output (MIMO) non-terrestrial network (NTN) that targets the downlink sum-rate maximization under imperfect local channel state information (CSI). We propose a novel entropy-based multi-agent deep reinforcement learning (DRL) approach where each non-terrestrial base station (NTBS) independently computes its beamforming vector using a Fourier Neural Operator (FNO) to capture long-range dependencies in the frequency domain. To ensure scalability and robustness, the proposed framework integrates transfer learning based on a conjugate prior mechanism and a low-rank decomposition (LRD) technique, thus enabling efficient support for large-scale user deployments and aerial layers. Our simulation results demonstrate the superiority of the proposed method over baseline schemes including WMMSE, ZF, MRT, CNN-based DRL, and the deep deterministic policy gradient (DDPG) method in terms of average sum rate, robustness to CSI imperfection, user mobility, and scalability across varying network sizes and user densities. Furthermore, we show that the proposed method achieves significant computational efficiency compared to CNN-based and WMMSE methods, while reducing communication overhead in comparison with shared-critic DRL approaches.

---

## 47. 论文ID: 2512.23813v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23813v1.json'

---

## 48. BATISNet: Instance Segmentation of Tooth Point Clouds with Boundary Awareness

**论文链接:** [http://arxiv.org/abs/2512.24201v1](http://arxiv.org/abs/2512.24201v1)

**作者:** Yating Cai, Yanghui Xu, Zehua Hu, Jiazhou Chen, Jing Huang

**发布时间:** 2025-12-30

**备注:** 10 pages, 4 figures

### GPT解析

### 总结

本文提出了一种名为BATISNet的边界感知实例网络，用于牙齿点云分割，解决了现有语义分割方法在处理紧密排列、边界不清及复杂牙科病例时的局限性。

### 背景

牙齿点云分割对于诊断、临床辅助和治疗计划具有重要意义。现有方法多采用语义分割，关注不同类型牙齿间的语义特征，但由于牙齿结构紧密、边界不清，以及缺牙、错位牙等复杂情况，语义分割在处理复杂病例时效果不佳。

### 目的

解决现有语义分割方法在处理复杂牙科病例时的局限性，提高牙齿点云分割的准确性和鲁棒性。

### 方法

提出BATISNet网络模型，由特征提取主干和实例分割模块组成，既关注不同类型牙齿的语义特征提取，又学习单个牙齿的实例特征。同时设计了边界感知损失函数，专门监督实例间的边界分割，有效缓解牙齿粘连和边界模糊问题。

### 主要发现

广泛的实验结果表明，BATISNet在牙齿完整性分割方面优于现有方法。

### 结论

BATISNet为实际临床应用提供了更可靠、详细的数据支持。

### 翻译

准确的牙齿点云分割对于诊断、临床辅助和治疗计划具有重要意义。现有方法大多采用语义分割，关注不同类型牙齿之间的语义特征。然而，由于牙齿结构紧密、边界不清，以及缺牙、错位牙等复杂情况的多样性，语义分割在处理复杂牙科病例时往往难以获得满意的结果。为解决这些问题，本文提出了BATISNet，一种边界感知的牙齿点云分割实例网络。该网络模型由特征提取主干和实例分割模块组成。它不仅关注提取不同类型牙齿的语义特征，还学习单个牙齿的实例特征。这有助于在缺牙和错位牙等复杂临床场景中实现更鲁棒、准确的牙齿实例分割。此外，为了进一步提高牙齿边界分割的完整性和准确性，设计了一种边界感知损失函数，专门监督实例之间的边界分割。它有效缓解了牙齿粘连和边界模糊问题。大量实验结果表明，BATISNet在牙齿完整性分割方面优于现有方法，为实际临床应用提供了更可靠、详细的数据支持。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决口腔扫描模型中牙齿实例分割的准确性问题，特别是在处理缺失牙齿、错位牙齿和紧密排列等复杂临床案例时的分割挑战。这个问题在现实中非常重要，因为精确的牙齿分割是数字牙科的基础，能为正畸学、修复学和种植学等领域的临床决策和治疗规划提供关键数据支持，直接影响治疗效果和患者预后。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有语义分割方法在处理复杂牙科案例时的局限性，然后将牙齿分割任务重新定义为实例分割任务以提高灵活性。设计上借鉴了PointMLP网络作为特征提取主干，参考了2D图像实例分割方法(如OneFormer和Mask2Former)的Transformer解码器设计，并改进了边界检测和分割的相关研究。此外，还引入了Graph Cut后处理技术进一步优化分割结果。这些现有方法被作者进行了针对性的改进和适应，使其更适合牙齿点云分割的特殊需求。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将牙齿分割从语义分割重新定义为实例分割，同时关注牙齿的语义特征和实例特征，并特别强调边界的准确性。整体流程包括：1)输入点云数据(坐标和法线向量)；2)使用基于PointMLP的U-Net主干提取特征；3)通过掩模感知实例分割模块生成实例嵌入；4)应用边界感知损失函数优化边界分割；5)使用Graph Cut后处理进一步优化结果；6)输出最终的牙齿实例分割结果。这种方法能同时处理局部细节和全局语义信息，有效解决牙齿粘连和边界模糊问题。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)将牙齿分割重新定义为实例分割任务，提高处理复杂案例的灵活性；2)设计实例感知掩模嵌入机制，整合局部和全局特征；3)引入边界感知损失函数，专门优化边界分割；4)设计全局-局部特征聚合模块增强特征表示能力。相比之前工作，不同之处在于：不依赖预定义的固定牙齿类别标签，能处理可变数量的牙齿实例；不依赖初步预测(如边界框)来指导分割；边界损失函数提供更全面的边界检测解决方案；在复杂临床案例中表现更优越。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'BATISNet通过将牙齿分割重新定义为实例分割任务并引入边界感知机制，显著提高了口腔扫描模型中牙齿分割的准确性和完整性，特别是在处理复杂临床案例时表现出卓越性能。'}


### 论文摘要

Accurate segmentation of the tooth point cloud is of great significance for diagnosis clinical assisting and treatment planning. Existing methods mostly employ semantic segmentation, focusing on the semantic feature between different types of teeth. However, due to the tightly packed structure of teeth, unclear boundaries, and the diversity of complex cases such as missing teeth, malposed teeth, semantic segmentation often struggles to achieve satisfactory results when dealing with complex dental cases. To address these issues, this paper propose BATISNet, a boundary-aware instance network for tooth point cloud segmentation. This network model consists of a feature extraction backbone and an instance segmentation module. It not only focuses on extracting the semantic features of different types of teeth but also learns the instance features of individual teeth. It helps achieve more robust and accurate tooth instance segmentation in complex clinical scenarios such as missing teeth and malposed teeth. Additionally, to further enhance the completeness and accuracy of tooth boundary segmentation, a boundary-aware loss function is designed to specifically supervise the boundary segmentation between instances. It mitigates effectively tooth adhesion and boundary ambiguity issues. Extensive experimental results show that BATISNet outperforms existing methods in tooth integrity segmentation, providing more reliable and detailed data support for practical clinical applications.

---

## 49. 论文ID: 2512.24679v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.24679v1.json'

---

## 50. Collaborative Low-Rank Adaptation for Pre-Trained Vision Transformers

**论文链接:** [http://arxiv.org/abs/2512.24603v1](http://arxiv.org/abs/2512.24603v1)

**作者:** Zheng Liu, Jinchao Zhu, Gao Huang

**发布时间:** 2025-12-31

**备注:** 13 tables, 3 figures

### GPT解析

### 总结

这篇论文提出了协作低秩适应(CLoRA)方法，通过基空间共享和样本无关多样性增强(SADE)组件，解决了现有LoRA方法在学习性能和参数效率之间的平衡问题。

### 背景

现有的LoRA研究主要关注参数效率更高的策略或更有效的表示学习方案，但这些方法要么牺牲了微调性能，要么引入了过多的可训练参数，无法在学习性能和参数效率之间取得平衡。

### 目的

提出一种新的微调方法CLoRA，以解决现有方法在学习性能和参数效率之间的平衡问题。

### 方法

CLoRA由基空间共享和样本无关多样性增强(SADE)组件组成。基空间共享允许所有低秩模块共享一组下/上投影空间，在保持参数效率的同时扩展学习能力；SADE则正则化低秩矩阵之间的相似性，鼓励训练过程中多样化的表示。

### 主要发现

在广泛使用的图像和点云数据集上的实验表明，与最先进的方法相比，CLoRA在学习性能和参数效率之间取得了更好的平衡，并且在点云分析中需要最少的GFLOPs。

### 结论

CLoRA是一种有效的微调方法，能够在学习性能和参数效率之间取得良好平衡，特别适用于点云分析任务。

### 翻译

低秩适应(LoRA)在微调预训练视觉变压器以完成各种下游任务方面取得了显著成功。现有研究主要关注探索更参数高效的策略或更有效的表示学习方案。然而，这些方法要么牺牲了微调性能，要么引入了过多的可训练参数，无法在学习性能和参数效率之间取得平衡。为了解决这个问题，本文提出了一种名为协作低秩适应(CLoRA)的新型微调方法。CLoRA由基空间共享和样本无关多样性增强(SADE)组件组成。为了在保持参数效率的同时扩展低秩模块(LRMs)的学习能力，基空间共享允许所有LRMs共享一组下/上投影空间。在CLoRA中，从共享空间获得的低秩矩阵协作构建每个LRM。由于这些矩阵提取的表示可能包含冗余信息，SADE被用来正则化它们之间的相似性，以鼓励训练过程中多样化的表示。我们在广泛使用的图像和点云数据集上进行了大量实验来评估CLoRA的性能。实验结果表明，与最先进的方法相比，CLoRA在学习性能和参数效率之间取得了更好的平衡，同时在点云分析中需要最少的GFLOPs。


### 论文摘要

Low-rank adaptation (LoRA) has achieved remarkable success in fine-tuning pre-trained vision transformers for various downstream tasks. Existing studies mainly focus on exploring more parameter-efficient strategies or more effective representation learning schemes. However, these methods either sacrifice fine-tuning performance or introduce excessive trainable parameters, failing to strike a balance between learning performance and parameter efficiency. To address this problem, we propose a novel tuning method named collaborative low-rank adaptation (CLoRA) in this paper. CLoRA consists of base-space sharing and sample-agnostic diversity enhancement (SADE) components. To maintain parameter efficiency while expanding the learning capacity of low-rank modules (LRMs), base-space sharing allows all LRMs to share a set of down/up-projection spaces. In CLoRA, the low-rank matrices obtained from the shared spaces collaboratively construct each LRM. Since the representations extracted by these matrices may contain redundant information, SADE is employed to regularize the similarities among them to encourage diverse representations in the training process. We conduct extensive experiments on widely used image and point cloud datasets to evaluate the performance of CLoRA. Experimental results demonstrate that CLoRA strikes a better balance between learning performance and parameter efficiency, while requiring the fewest GFLOPs for point cloud analysis, compared with the state-of-the-art methods.

---

## 51. CPR: Causal Physiological Representation Learning for Robust ECG Analysis under Distribution Shifts

**论文链接:** [http://arxiv.org/abs/2512.24564v1](http://arxiv.org/abs/2512.24564v1)

**作者:** Shunbo Jia, Caizhi Liao

**发布时间:** 2025-12-31

### GPT解析

### 总结

该研究提出了一种名为'因果生理表征学习'(CPR)的新方法，用于增强心电图(ECG)诊断深度学习模型对抗对抗性扰动的能力，特别是模拟生物形态的平滑对抗性扰动(SAP)。该方法通过在因果解缠框架中整合生理结构先验，实现了在不增加计算负担的情况下保持模型鲁棒性和效率。

### 背景

深度学习模型在心电图诊断中已取得显著准确性，但对对抗性扰动（特别是模拟生物形态的平滑对抗性扰动）表现出脆弱性。现有防御面临两难困境：对抗训练(AT)提供鲁棒性但计算负担沉重，而认证方法如随机平滑(RS)则引入显著推理延迟，使其不适用于实时临床监测。

### 目的

解决心电图诊断模型对抗对抗性扰动的脆弱性问题，特别关注平滑对抗性扰动(SAP)。研究旨在开发一种方法，既能提供与现有防御方法相当的鲁棒性，又能保持计算效率和临床实用性。

### 方法

研究提出'因果生理表征学习'(CPR)方法。与没有语义约束的标准去噪方法不同，CPR在因果解缠框架中整合了生理结构先验。通过将心电图生成建模为结构因果模型(SCM)，CPR强制执行结构干预，将不变的病理形态(P-QRS-T复合物)与非因果性人工制品严格分离。

### 主要发现

在PTB-XL数据集上的实证结果表明，CPR显著优于标准临床预处理方法。具体而言，在SAP攻击下，CPR达到0.632的F1分数，比中值平滑(0.541 F1)高出9.1%。关键的是，CPR在保持单次推理效率的同时，匹配了随机平滑的认证鲁棒性，提供了鲁棒性、效率和临床可解释性之间的更好权衡。

### 结论

CPR方法成功解决了心电图诊断模型对抗对抗性扰动的脆弱性问题，通过在因果解缠框架中整合生理结构先验，实现了在不增加计算负担的情况下保持模型鲁棒性和效率，为临床应用提供了更好的解决方案。

### 翻译

心电图(ECG)诊断的深度学习模型已取得显著准确性，但对对抗性扰动表现出脆弱性，特别是模拟生物形态的平滑对抗性扰动(SAP)。现有防御面临关键困境：对抗训练(AT)提供鲁棒性但计算负担沉重，而像随机平滑(RS)这样的认证方法则引入显著的推理延迟，使其不适用于实时临床监测。我们认为这种脆弱性源于模型依赖于非鲁棒的虚假相关性，而非不变的病理特征。为此，我们提出了因果生理表征学习(CPR)。与没有语义约束的标准去噪方法不同，CPR在因果解缠框架中整合了生理结构先验。通过通过结构因果模型(SCM)对心电图生成进行建模，CPR强制执行结构干预，严格将不变的病理形态(P-QRS-T复合物)与非因果性人工制品分离。PTB-XL上的实证结果表明，CPR显著优于标准临床预处理方法。具体而言，在SAP攻击下，CPR达到0.632的F1分数，比中值平滑(0.541 F1)高出9.1%。重要的是，CPR在保持单次推理效率的同时匹配了随机平滑的认证鲁棒性，提供了鲁棒性、效率和临床可解释性之间的优越权衡。


### 论文摘要

Deep learning models for Electrocardiogram (ECG) diagnosis have achieved remarkable accuracy but exhibit fragility against adversarial perturbations, particularly Smooth Adversarial Perturbations (SAP) that mimic biological morphology. Existing defenses face a critical dilemma: Adversarial Training (AT) provides robustness but incurs a prohibitive computational burden, while certified methods like Randomized Smoothing (RS) introduce significant inference latency, rendering them impractical for real-time clinical monitoring. We posit that this vulnerability stems from the models' reliance on non-robust spurious correlations rather than invariant pathological features. To address this, we propose Causal Physiological Representation Learning (CPR). Unlike standard denoising approaches that operate without semantic constraints, CPR incorporates a Physiological Structural Prior within a causal disentanglement framework. By modeling ECG generation via a Structural Causal Model (SCM), CPR enforces a structural intervention that strictly separates invariant pathological morphology (P-QRS-T complex) from non-causal artifacts. Empirical results on PTB-XL demonstrate that CPR significantly outperforms standard clinical preprocessing methods. Specifically, under SAP attacks, CPR achieves an F1 score of 0.632, surpassing Median Smoothing (0.541 F1) by 9.1%. Crucially, CPR matches the certified robustness of Randomized Smoothing while maintaining single-pass inference efficiency, offering a superior trade-off between robustness, efficiency, and clinical interpretability.

---

## 52. Exploring Compositionality in Vision Transformers using Wavelet Representations

**论文链接:** [http://arxiv.org/abs/2512.24438v1](http://arxiv.org/abs/2512.24438v1)

**作者:** Akshad Shyam Purushottamdas, Pranav K Nayak, Divya Mehul Rajparia, Deekshith Patel, Yashmitha Gogineni, Konda Reddy Mopuri, Sumohana S. Channappayya

**发布时间:** 2025-12-30

**备注:** 9 pages, 6 figures

### GPT解析

### 总结

本研究通过组合性视角分析Vision Transformer (ViT)编码器学习到的表示，引入一个类似于先前表示学习中测量组合性的框架，利用离散小波变换(DWT)获取输入相关的基元，通过检查组合表示重现原始图像表示的能力，经验性测试表示空间中的组合性程度。

### 背景

大多数关于transformer模型的见解是通过分析它们在语言任务上的行为获得的，而ViT表示的组合性研究相对较少。

### 目的

引入一个框架测试ViT编码器中的组合性，理解ViT如何结构化信息。

### 方法

使用离散小波变换(DWT)作为工具获取输入相关的基元，通过检查组合表示重现原始图像表示的能力，经验性测试表示空间中的组合性程度。

### 主要发现

来自一级DWT分解的基元产生的编码器表示在潜在空间中大约组合，表明ViT的表示空间中存在组合性。

### 结论

ViT的表示空间中存在组合性，这为理解ViT如何结构化信息提供了新的视角。

### 翻译

虽然对transformer模型工作原理的见解大多是通过分析它们在语言任务上的行为而产生的，但这项工作通过组合性的视角研究了Vision Transformer (ViT)编码器学习到的表示。我们引入了一个框架，类似于先前在表示学习中测量组合性的工作，用于测试ViT编码器中的组合性。建立这种类别的关键是离散小波变换(DWT)，这是一种简单而有效的工具，用于在视觉设置中获取输入相关的基元。通过检查组合表示重现原始图像表示的能力，我们经验性地测试了表示空间中组合性的程度。我们的研究表明，来自一级DWT分解的基元产生的编码器表示在潜在空间中大约组合，这为ViT如何结构化信息提供了新的视角。


### 论文摘要

While insights into the workings of the transformer model have largely emerged by analysing their behaviour on language tasks, this work investigates the representations learnt by the Vision Transformer (ViT) encoder through the lens of compositionality. We introduce a framework, analogous to prior work on measuring compositionality in representation learning, to test for compositionality in the ViT encoder. Crucial to drawing this analogy is the Discrete Wavelet Transform (DWT), which is a simple yet effective tool for obtaining input-dependent primitives in the vision setting. By examining the ability of composed representations to reproduce original image representations, we empirically test the extent to which compositionality is respected in the representation space. Our findings show that primitives from a one-level DWT decomposition produce encoder representations that approximately compose in latent space, offering a new perspective on how ViTs structure information.

---

## 53. Hyperspherical Graph Representation Learning via Adaptive Neighbor-Mean Alignment and Uniformity

**论文链接:** [http://arxiv.org/abs/2512.24062v1](http://arxiv.org/abs/2512.24062v1)

**作者:** Rui Chen, Junjun Guo, Hongbin Wang, Yan Xiang, Yantuan Xian, Zhengtao Yu

**发布时间:** 2025-12-30

**备注:** Submitted to Pattern Recognition

### GPT解析

### 总结

HyperGRL是一种超球面图表示学习框架，通过自适应邻居均值对齐和无采样均匀性实现，无需复杂架构和负采样策略，在多种图任务上表现优异。

### 背景

图表示学习(GRL)旨在将图结构数据的结构和语义依赖关系编码为低维嵌入，但现有方法通常依赖代理对比目标或互信息最大化，需要复杂架构、负采样策略和敏感的超参数调整。

### 目的

提出HyperGRL框架，解决现有GRL方法中过度平滑、过度挤压和训练不稳定问题，实现更稳定高效的图表示学习。

### 方法

HyperGRL通过两个对抗耦合目标将节点嵌入到单位超球面上：1)邻居均值对齐，使用节点局部邻域的均值表示构建语义基础稳定的目标；2)无采样均匀性，通过基于L2的超球面正则化鼓励全局均匀嵌入分布；3)引入熵引导的自适应平衡机制动态调节两个目标间的相互作用。

### 主要发现

在节点分类、节点聚类和链接预测任务上，HyperGRL比现有最强方法分别平均提高1.49%、0.86%和0.74%，展示了基于几何的、无采样的对比目标对于图表示学习的有效性。

### 结论

HyperGRL通过自适应邻居均值对齐和无采样均匀性，实现了更稳定、高效的图表示学习，在多种图结构和任务上表现优异。

### 翻译

图表示学习(GRL)旨在将图结构数据的结构和语义依赖关系编码为低维嵌入。然而，现有的GRL方法通常依赖于代理对比目标或互信息最大化，这些方法通常需要复杂的架构、负采样策略和敏感的超参数调整。这些设计选择可能导致过度平滑、过度挤压和训练不稳定。在本工作中，我们提出了HyperGRL，一个通过自适应邻居均值对齐和无采样均匀性实现的超球面图表示学习的统一框架。HyperGRL通过两个对抗耦合目标将节点嵌入到单位超球面上：邻居均值对齐和无采样均匀性。对齐目标使用每个节点局部邻域的均值表示来构建语义基础稳定的目标，捕获共享的结构和特征模式。均匀性目标通过基于L2的超球面正则化公式化离散度，鼓励全局均匀的嵌入分布，同时保持判别信息。为了进一步稳定训练，我们引入了一个熵引导的自适应平衡机制，动态调节对齐和均匀性之间的相互作用，无需手动调整。在节点分类、节点聚类和链接预测上的广泛实验表明，HyperGRL在多样化的图结构上提供卓越的表示质量和泛化能力，分别比现有最强方法平均提高1.49%、0.86%和0.74%。这些发现强调了基于几何的、无采样的对比目标对于图表示学习的有效性。


### 论文摘要

Graph representation learning (GRL) aims to encode structural and semantic dependencies of graph-structured data into low-dimensional embeddings. However, existing GRL methods often rely on surrogate contrastive objectives or mutual information maximization, which typically demand complex architectures, negative sampling strategies, and sensitive hyperparameter tuning. These design choices may induce over-smoothing, over-squashing, and training instability. In this work, we propose HyperGRL, a unified framework for hyperspherical graph representation learning via adaptive neighbor-mean alignment and sampling-free uniformity. HyperGRL embeds nodes on a unit hypersphere through two adversarially coupled objectives: neighbor-mean alignment and sampling-free uniformity. The alignment objective uses the mean representation of each node's local neighborhood to construct semantically grounded, stable targets that capture shared structural and feature patterns. The uniformity objective formulates dispersion via an L2-based hyperspherical regularization, encouraging globally uniform embedding distributions while preserving discriminative information. To further stabilize training, we introduce an entropy-guided adaptive balancing mechanism that dynamically regulates the interplay between alignment and uniformity without requiring manual tuning. Extensive experiments on node classification, node clustering, and link prediction demonstrate that HyperGRL delivers superior representation quality and generalization across diverse graph structures, achieving average improvements of 1.49%, 0.86%, and 0.74% over the strongest existing methods, respectively. These findings highlight the effectiveness of geometrically grounded, sampling-free contrastive objectives for graph representation learning.

---

## 54. Disentangling Learning from Judgment: Representation Learning for Open Response Analytics

**论文链接:** [http://arxiv.org/abs/2512.23941v1](http://arxiv.org/abs/2512.23941v1)

**作者:** Conrad Borchers, Manit Patel, Seiyon M. Lee, Anthony F. Botelho

**发布时间:** 2025-12-30

**DOI:** 10.1145/3785022.3785042

**备注:** Short research paper accepted at Learning Analytics and Knowledge (LAK '26)

### GPT解析

### 总结

该研究提出了一个分析框架，将内容信号与评分者倾向分离，通过分析技术使评分判断可见且可审核。研究使用动态先验和文本表示方法，结合中心化和残差化技术减少混淆因素，结果显示教师先验对评分预测有重大影响，结合内容嵌入时效果最佳。

### 背景

开放式回答在学习中至关重要，但自动评分常常将学生写的内容与教师评分方式混淆，缺乏清晰区分内容质量和评分者偏好的方法。

### 目的

开发一个分析优先的框架，将内容信号与评分者倾向分离开，使评分判断通过分析技术变得可见和可审核，帮助教师和研究者审视评分实践与学生推理和学习证据的一致性。

### 方法

使用去标识化的ASSISTments数学回答，将教师历史建模为动态先验，从句子嵌入中推导文本表示，采用中心化和残差化减少提示和评分者混淆。通过时间验证的线性模型量化每个信号的贡献，并通过投影展示模型分歧以进行定性检查。

### 主要发现

教师先验对评分预测有重大影响；当先验与内容嵌入结合时效果最佳（AUC约0.815），而仅使用内容的模型虽然高于随机水平但明显较弱（AUC约0.626）。调整评分者效应后，残差内容表示保留了更多信息丰富的嵌入维度，揭示了语义证据支持理解的案例，而非表面回答差异。

### 结论

该研究提出了一个实用的流程，将嵌入从简单特征转变为用于反思的学习分析工具，使教师和研究者能够检查评分实践与学生推理和学习证据的一致性或冲突。

### 翻译

开放式回答是学习的核心，但自动评分常常将学生所写内容与教师评分方式混为一谈。我们提出了一个分析优先的框架，将内容信号与评分者倾向分离，通过分析技术使判断变得可见和可审核。使用去标识化的ASSISTments数学回答，我们将教师历史建模为动态先验，并从句子嵌入中推导文本表示，采用中心化和残差化来减少提示和评分者混淆。通过时间验证的线性模型量化每个信号的贡献，并通过投影展示模型分歧以进行定性检查。结果表明，教师先验对评分预测有重大影响；当先验与内容嵌入结合时效果最佳（AUC约0.815），而仅使用内容的模型虽然高于随机水平但明显较弱（AUC约0.626）。调整评分者效应后，残差内容表示保留了更多信息丰富的嵌入维度，揭示了语义证据支持理解的案例，而不是学生回答方式的表面差异。该研究贡献了一个实用的流程，将嵌入从简单特征转变为用于反思的学习分析工具，使教师和研究者能够检查评分实践与学生推理和学习证据的一致性或冲突。


### 论文摘要

Open-ended responses are central to learning, yet automated scoring often conflates what students wrote with how teachers grade. We present an analytics-first framework that separates content signals from rater tendencies, making judgments visible and auditable via analytics. Using de-identified ASSISTments mathematics responses, we model teacher histories as dynamic priors and derive text representations from sentence embeddings, incorporating centering and residualization to mitigate prompt and teacher confounds. Temporally-validated linear models quantify the contributions of each signal, and a projection surfaces model disagreements for qualitative inspection. Results show that teacher priors heavily influence grade predictions; the strongest results arise when priors are combined with content embeddings (AUC~0.815), while content-only models remain above chance but substantially weaker (AUC~0.626). Adjusting for rater effects sharpens the residual content representation, retaining more informative embedding dimensions and revealing cases where semantic evidence supports understanding as opposed to surface-level differences in how students respond. The contribution presents a practical pipeline that transforms embeddings from mere features into learning analytics for reflection, enabling teachers and researchers to examine where grading practices align (or conflict) with evidence of student reasoning and learning.

---

## 55. Visual Language Hypothesis

**论文链接:** [http://arxiv.org/abs/2512.23335v2](http://arxiv.org/abs/2512.23335v2)

**作者:** Xiu Li

**发布时间:** 2025-12-29

### GPT解析

### 总结

研究从结构和拓扑角度探索视觉表征学习，提出视觉理解需要视觉语义语言的理论框架

### 背景

视觉表征学习领域，关注如何从视觉数据中提取有意义的信息

### 目的

探索视觉表征学习的结构和拓扑特性，理解视觉语义的形成机制

### 方法

基于纤维束状结构理论，分析视觉观测空间的组织方式，推导理论推论

### 主要发现

语义商空间不能通过光滑变形获得，需要非同胚、判别性目标；语义抽象需要支持拓扑变化的表征机制，包括扩展和捕捉过程

### 结论

该框架提供了一种拓扑视角，与大规模判别性和多模态模型的经验规律一致，也与经典统计学习理论原则相符

### 翻译

我们从结构和拓扑角度研究视觉表征学习。我们从一个单一假设出发：视觉理解 presupposes 一种视觉语义语言，其中许多感知观测对应少数离散语义状态。结合表征学习中普遍假设的可转移性和抽象性前提，这一假设意味着视觉观测空间必须以纤维束状结构组织，其中 nuisance 变量填充纤维，语义对应商基空间。从这一结构我们推导出两个理论推论。首先，语义商 X/G 不是 X 的子流形，不能仅通过光滑变形获得，语义不变性需要非同胚、判别性目标，例如通过标签的监督、跨实例识别或多模态对齐提供显式语义等价。其次，我们表明近似商空间也对模型架构提出了结构要求。语义抽象不仅需要外部语义目标，还需要能够支持拓扑变化的表征机制：一个扩展和捕捉过程，流形首先几何扩展以分离结构，然后折叠形成离散语义区域。我们强调这些结果是解释性的而非规定性的：该框架提供了一个与大规模判别性和多模态模型观察到的经验规律一致的拓扑视角，也与统计学习理论中的经典原则一致


### 论文摘要

We study visual representation learning from a structural and topological perspective. We begin from a single hypothesis: that visual understanding presupposes a semantic language for vision, in which many perceptual observations correspond to a small number of discrete semantic states. Together with widely assumed premises on transferability and abstraction in representation learning, this hypothesis implies that the visual observation space must be organized in a fiber bundle like structure, where nuisance variation populates fibers and semantics correspond to a quotient base space. From this structure we derive two theoretical consequences. First, the semantic quotient X/G is not a submanifold of X and cannot be obtained through smooth deformation alone, semantic invariance requires a non homeomorphic, discriminative target for example, supervision via labels, cross-instance identification, or multimodal alignment that supplies explicit semantic equivalence. Second, we show that approximating the quotient also places structural demands on the model architecture. Semantic abstraction requires not only an external semantic target, but a representation mechanism capable of supporting topology change: an expand and snap process in which the manifold is first geometrically expanded to separate structure and then collapsed to form discrete semantic regions. We emphasize that these results are interpretive rather than prescriptive: the framework provides a topological lens that aligns with empirical regularities observed in large-scale discriminative and multimodal models, and with classical principles in statistical learning theory.

---

## 56. Tracking by Predicting 3-D Gaussians Over Time

**论文链接:** [http://arxiv.org/abs/2512.22489v2](http://arxiv.org/abs/2512.22489v2)

**作者:** Tanish Baranwal, Himanshu Gaurav Singh, Jathushan Rajasegaran, Jitendra Malik

**发布时间:** 2025-12-27

### GPT解析

### 总结

本文提出Video Gaussian Masked Autoencoders (Video-GMAE)，一种自监督表征学习方法，将视频序列编码为随时间移动的高斯斑点集合。该方法利用三维场景投影的归纳偏置，在零样本跟踪任务上达到与最先进方法相当的性能，并在经过微调后在Kinetics和Kubric数据集上显著超越现有方法。

### 背景

视频表征学习是计算机视觉的重要课题，现有的自监督视频方法仍有提升空间。视频作为三维场景的二维投影，具有内在的三维一致性特性，可作为合理的归纳偏置。

### 目的

开发一种新的自监督视频表征学习方法，有效捕捉视频中的时空信息，利用三维场景投影特性作为归纳偏置，提升视频表征学习和跟踪任务性能。

### 方法

提出Video-GMAE架构，将视频序列编码为随时间移动的高斯斑点集合，强制网络学习视频作为动态三维场景投影的特性。通过预训练这种架构，使跟踪能力自然涌现。

### 主要发现

1) 高斯斑点表示强制执行了视频作为三维场景投影的合理归纳偏置；2) 预训练后网络自然具备跟踪能力；3) 将高斯轨迹映射到图像平面实现零样本跟踪；4) 微调后在Kinetics和Kubric数据集上分别实现34.6%和13.1%的性能提升。

### 结论

Video-GMAE是一种有效的自监督视频表征学习方法，通过高斯斑点表示和三维投影归纳偏置，在视频表征学习和跟踪任务上均取得优异性能，超越现有自监督视频方法。

### 翻译

我们提出Video Gaussian Masked Autoencoders (Video-GMAE)，一种用于表征学习的自监督方法，它将图像序列编码为随时间移动的高斯斑点集合。将视频表示为高斯斑点集合强制执行了合理的归纳偏置：二维视频通常是动态三维场景的一致投影。我们发现，使用这种架构预训练网络时，跟踪能力 emerges。将学习到的高斯轨迹映射到图像平面上，可以实现与最先进方法相当的零样本跟踪性能。经过小规模微调，我们的模型在Kinetics数据集上实现了34.6%的改进，在Kubric数据集上实现了13.1%的改进，超越了现有的自监督视频方法。项目页面和代码可在https://videogmae.org/和https://github.com/tekotan/video-gmae公开获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视频中的点跟踪问题，即如何在视频中持续跟踪特定像素点的位置。这个问题很重要，因为点跟踪是理解视频内容、场景结构和对象交互的基础能力，对于计算摄影、3D理解和其他需要长程推理的任务都至关重要。现有的点跟踪方法大多依赖大量标注数据或特定架构，限制了应用范围，而自监督学习可以减少对昂贵人工标注的依赖。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者发现现有视频自监督学习在点跟踪上表现不佳，假设传统方法不能强制执行时间一致性。他们提出关键见解：3D物体在3D空间中的运动表现为图像平面上的点跟踪。基于此，他们设计了一个掩码自编码器风格的架构，借鉴了BERT的掩码建模思想、3D高斯溅射技术以及视频自监督学习方法，创新性地将它们结合，通过预测随时间变化的高斯基元来强制执行时间对应关系。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将视频序列表示为一组随时间移动的3D高斯基元，通过预测高斯基元随时间的变化来强制执行时间对应关系，施加2D视频是动态3D场景一致投影的归纳偏置。整体流程是：输入视频序列→编码器处理被遮蔽的帧→解码器生成高斯基元(第一帧预测完整高斯，后续帧只预测变化量)→使用高斯溅射技术渲染→通过重建损失训练→从高斯轨迹计算点轨迹实现零样本跟踪。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：提出Video-GMAE新方法，通过预测高斯时间变化强制执行对应关系；首次展示自监督预训练中涌现零样本点跟踪能力；设计从高斯轨迹计算点轨迹的算法；在多个数据集实现最先进跟踪性能。相比之前工作，不同在于：显式建模3D空间中的时间对应关系，而非传统方法的2D块预测；不需要跟踪标注数据；使用3D高斯表示而非2D特征；将高斯溅射扩展到视频表示学习而非仅用于3D重建。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了Video-GMAE，一种通过预测随时间变化的3D高斯基元进行自监督视频表示学习的方法，实现了强大的零样本点跟踪能力，并在多个数据集上取得了最先进的跟踪性能。'}


### 论文摘要

We propose Video Gaussian Masked Autoencoders (Video-GMAE), a self-supervised approach for representation learning that encodes a sequence of images into a set of Gaussian splats moving over time. Representing a video as a set of Gaussians enforces a reasonable inductive bias: that 2-D videos are often consistent projections of a dynamic 3-D scene. We find that tracking emerges when pretraining a network with this architecture. Mapping the trajectory of the learnt Gaussians onto the image plane gives zero-shot tracking performance comparable to state-of-the-art. With small-scale finetuning, our models achieve 34.6% improvement on Kinetics, and 13.1% on Kubric datasets, surpassing existing self-supervised video approaches. The project page and code are publicly available at https://videogmae.org/ and https://github.com/tekotan/video-gmae.

---

## 57. AudioFab: Building A General and Intelligent Audio Factory through Tool Learning

**论文链接:** [http://arxiv.org/abs/2512.24645v1](http://arxiv.org/abs/2512.24645v1)

**作者:** Cheng Zhu, Jing Han, Qianshuai Xue, Kehan Wang, Huan Zhao, Zixing Zhang

**发布时间:** 2025-12-31

**DOI:** 10.1145/3746027.3756869

### GPT解析

### 总结

AudioFab是一个开源的音频智能体框架，旨在解决现有音频AI工具分散、环境配置复杂和工具协作效率低的问题，为音频和多模态AI研究提供稳定可扩展的平台。

### 背景

人工智能正在深刻改变音频领域，但许多先进算法和工具仍然分散，缺乏统一高效的框架来释放它们的全部潜力。现有音频智能体框架通常存在复杂的环境配置和低效的工具协作问题。

### 目的

为了解决这些限制，引入AudioFab，建立开放和智能的音频处理生态系统，简化工具集成和扩展，提高音频任务处理效率和准确性。

### 方法

采用模块化设计解决依赖冲突，简化工具集成和扩展；通过智能选择和少样本学习优化工具学习；为非专业用户提供用户友好的自然语言界面。

### 主要发现

AudioFab的模块化设计有效解决了依赖冲突问题，智能学习机制提高了复杂音频任务的效率和准确性，自然语言界面降低了使用门槛。

### 结论

AudioFab作为基础框架，为音频和多模态AI的未来研究和发展提供了稳定且可扩展的平台，其开源代码可在GitHub获取。

### 翻译

目前，人工智能正在深刻改变音频领域；然而，许多先进算法和工具仍然分散，缺乏统一高效的框架来释放它们的全部潜力。现有的音频智能体框架通常存在复杂的环境配置和低效的工具协作问题。为了解决这些限制，我们引入了AudioFab，这是一个开源的智能体框架，旨在建立一个开放和智能的音频处理生态系统。与现有解决方案相比，AudioFab的模块化设计解决了依赖冲突，简化了工具集成和扩展。它还通过智能选择和少样本学习优化工具学习，提高了复杂音频任务的效率和准确性。此外，AudioFab为非专业用户提供了用户友好的自然语言界面。作为基础框架，AudioFab的核心贡献在于为音频和多模态AI的未来研究和发展提供了稳定且可扩展的平台。代码可在https://github.com/SmileHnu/AudioFab获取。


### 论文摘要

Currently, artificial intelligence is profoundly transforming the audio domain; however, numerous advanced algorithms and tools remain fragmented, lacking a unified and efficient framework to unlock their full potential. Existing audio agent frameworks often suffer from complex environment configurations and inefficient tool collaboration. To address these limitations, we introduce AudioFab, an open-source agent framework aimed at establishing an open and intelligent audio-processing ecosystem. Compared to existing solutions, AudioFab's modular design resolves dependency conflicts, simplifying tool integration and extension. It also optimizes tool learning through intelligent selection and few-shot learning, improving efficiency and accuracy in complex audio tasks. Furthermore, AudioFab provides a user-friendly natural language interface tailored for non-expert users. As a foundational framework, AudioFab's core contribution lies in offering a stable and extensible platform for future research and development in audio and multimodal AI. The code is available at https://github.com/SmileHnu/AudioFab.

---

## 58. Adaptive Learning Guided by Bias-Noise-Alignment Diagnostics

**论文链接:** [http://arxiv.org/abs/2512.24445v1](http://arxiv.org/abs/2512.24445v1)

**作者:** Akash Samanta, Sheldon Williamson

**发布时间:** 2025-12-30

**备注:** This preprint focuses on the theoretical framework and diagnostic behavior. Comprehensive experimental validation in application-specific settings is deferred to a companion experimental study

### GPT解析

### 总结

本文提出了一种基于诊断的自适应学习框架，通过将误差信号分解为偏差、噪声和对齐三个组成部分，提高在非平稳和安全关键环境中的学习系统稳定性。该方法适用于监督优化、参与者-批评强化学习和学习型优化器，提供了可解释的、轻量级的基础。

### 背景

在非平稳和安全关键环境中部署的学习系统，当学习动态随时间演变时，常常遭受不稳定、收敛缓慢或适应性脆弱的问题。现代优化、强化学习和元学习方法虽然能够适应梯度统计特性，但 largely 忽略了误差信号本身的时序结构。

### 目的

提出一种诊断驱动的自适应学习框架，明确地对误差演变进行建模，将其分解为捕捉持续漂移的偏差、捕捉随机变化的噪声以及捕捉导致过冲的重复方向性激励的对齐。这些诊断是通过损失或时序差分误差轨迹的轻量级统计量在线计算的，并且独立于模型架构或任务领域。

### 方法

提出了一种偏差-噪声-对齐分解方法，为监督优化、参与者-批评强化学习和学习型优化器提供统一的控制主干。基于此框架，推导出诊断驱动的实例化方法，包括稳定的监督优化器、诊断调节的参与者-批评方案和诊断条件化的学习型优化器。

### 主要发现

在标准平滑性假设下，为所有情况建立了有界有效更新和稳定性特性。参与者-批评学习中的代表性诊断说明，所提出的信号如何根据时序差分误差结构调节适应性。

### 结论

这项工作将误差演变提升为自适应学习中的一等对象，为动态环境中的可靠学习提供了可解释的、轻量级的基础。

### 翻译

部署在非平稳和安全关键环境中的学习系统，当学习动态随时间演变时，常常遭受不稳定、收敛缓慢或适应性脆弱的问题。虽然现代优化、强化学习和元学习方法能够适应梯度统计特性，但它们 largely 忽略了误差信号本身的时序结构。本文提出了一种基于诊断的自适应学习框架，通过将误差信号明确分解为偏差（捕捉持续漂移）、噪声（捕捉随机变化）和对齐（捕捉导致过冲的重复方向性激励）来对误差演变进行建模。这些诊断是通过损失或时序差分误差轨迹的轻量级统计量在线计算的，并且独立于模型架构或任务领域。我们表明，所提出的偏差-噪声-对齐分解为监督优化、参与者-批评强化学习和学习型优化器提供了统一的控制主干。基于此框架，我们推导出诊断驱动的实例化方法，包括稳定的监督优化器、诊断调节的参与者-批评方案和诊断条件化的学习型优化器。在标准平滑性假设下，我们为所有情况建立了有界有效更新和稳定性特性。参与者-批评学习中的代表性诊断说明，所提出的信号如何根据时序差分误差结构调节适应性。总体而言，这项工作将误差演变提升为自适应学习中的一等对象，为动态环境中的可靠学习提供了可解释的、轻量级的基础。


### 论文摘要

Learning systems deployed in nonstationary and safety-critical environments often suffer from instability, slow convergence, or brittle adaptation when learning dynamics evolve over time. While modern optimization, reinforcement learning, and meta-learning methods adapt to gradient statistics, they largely ignore the temporal structure of the error signal itself. This paper proposes a diagnostic-driven adaptive learning framework that explicitly models error evolution through a principled decomposition into bias, capturing persistent drift; noise, capturing stochastic variability; and alignment, capturing repeated directional excitation leading to overshoot. These diagnostics are computed online from lightweight statistics of loss or temporal-difference error trajectories and are independent of model architecture or task domain. We show that the proposed bias-noise-alignment decomposition provides a unifying control backbone for supervised optimization, actor-critic reinforcement learning, and learned optimizers. Building on this framework, we derive diagnostic-driven instantiations including a stabilized supervised optimizer, a diagnostic-regulated actor-critic scheme, and a diagnostic-conditioned learned optimizer. Under standard smoothness assumptions, we establish bounded effective updates and stability properties for all cases. Representative diagnostic illustrations in actor-critic learning highlight how the proposed signals modulate adaptation in response to temporal-difference error structure. Overall, this work elevates error evolution to a first-class object in adaptive learning and provides an interpretable, lightweight foundation for reliable learning in dynamic environments.

---

## 59. Enhancing LLM Planning Capabilities through Intrinsic Self-Critique

**论文链接:** [http://arxiv.org/abs/2512.24103v1](http://arxiv.org/abs/2512.24103v1)

**作者:** Bernd Bohnet, Pierre-Alexandre Kamienny, Hanie Sedghi, Dilan Gorur, Pranjal Awasthi, Aaron Parisi, Kevin Swersky, Rosanne Liu, Azade Nova, Noah Fiedel

**发布时间:** 2025-12-30

### GPT解析

### 总结

这篇论文提出了一种让大语言模型自我批判以提高性能的方法，在多个规划数据集上取得了显著改进，超越了之前的基准和基线准确率。

### 背景

早期研究对大语言模型利用自我批判方法的有效性表示怀疑，需要探索更有效的自我改进方法。

### 目的

开发一种让大语言模型能够自我批判和自我改进的方法，以提高其在规划任务上的性能。

### 方法

采用内在自我批判方法，不需要外部验证器；使用少样本学习技术，逐步扩展为多样本方法作为基础方法；通过迭代过程进行修正和精炼。

### 主要发现

在Blocksworld领域的规划数据集上通过内在自我批判取得了显著性能提升；在Logistics和Mini-grid数据集上也展示了类似的改进，超过了强大的基线准确率；自我批判可以显著提高规划性能；该方法在2024年10月的LLM模型检查点上达到了新的最先进水平。

### 结论

自我批判是一种有效的自我改进方法，适用于任何特定模型版本；将该方法应用于更复杂的搜索技术和更强大的模型可能会带来更好的性能。

### 翻译

我们展示了一种让大语言模型批判自己答案的方法，目的是提高它们的性能，在既定的规划基准测试中取得了显著改进。尽管早期研究对大语言模型利用自我批判方法的有效性表示怀疑，但我们通过内在自我批判方法在Blocksworld领域的规划数据集上取得了显著的性能提升，不需要外部验证器等来源。我们还展示了在Logistics和Mini-grid数据集上的类似改进，超过了强大的基线准确率。我们采用少样本学习技术，并将其逐步扩展为多样本方法作为我们的基础方法，并证明通过采用迭代修正和精炼过程，可以在这个已经具有竞争力的方法基础上获得实质性改进。我们展示了自我批判如何显著提高规划性能。我们的实证结果在所考虑的模型类别上呈现了新的最先进水平，即2024年10月的大语言模型检查点。我们的主要关注点是方法本身，展示了内在自我改进能力，这些能力适用于任何特定模型版本，我们相信将该方法应用于更复杂的搜索技术和更强大的模型将带来更好的性能。


### 论文摘要

We demonstrate an approach for LLMs to critique their \emph{own} answers with the goal of enhancing their performance that leads to significant improvements over established planning benchmarks. Despite the findings of earlier research that has cast doubt on the effectiveness of LLMs leveraging self critique methods, we show significant performance gains on planning datasets in the Blocksworld domain through intrinsic self-critique, without external source such as a verifier. We also demonstrate similar improvements on Logistics and Mini-grid datasets, exceeding strong baseline accuracies. We employ a few-shot learning technique and progressively extend it to a many-shot approach as our base method and demonstrate that it is possible to gain substantial improvement on top of this already competitive approach by employing an iterative process for correction and refinement. We illustrate how self-critique can significantly boost planning performance. Our empirical results present new state-of-the-art on the class of models considered, namely LLM model checkpoints from October 2024. Our primary focus lies on the method itself, demonstrating intrinsic self-improvement capabilities that are applicable regardless of the specific model version, and we believe that applying our method to more complex search techniques and more capable models will lead to even better performance.

---

## 60. 论文ID: 2512.23987v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23987v1.json'

---

## 61. 论文ID: 2512.23848v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23848v1.json'

---

## 62. MiMo-Audio: Audio Language Models are Few-Shot Learners

**论文链接:** [http://arxiv.org/abs/2512.23808v1](http://arxiv.org/abs/2512.23808v1)

**作者:** Xiaomi LLM-Core Team, :, Dong Zhang, Gang Wang, Jinlong Xue, Kai Fang, Liang Zhao, Rui Ma, Shuhuai Ren, Shuo Liu, Tao Guo, Weiji Zhuang, Xin Zhang, Xingchen Song, Yihan Yan, Yongzhe He, Cici, Bowen Shen, Chengxuan Zhu, Chong Ma, Chun Chen, Heyu Chen, Jiawei Li, Lei Li, Menghang Zhu, Peidian Li, Qiying Wang, Sirui Deng, Weimin Xiong, Wenshan Huang, Wenyu Yang, Yilin Jiang, Yixin Yang, Yuanyuan Tian, Yue Ma, Yue Yu, Zihan Zhang, Zihao Yue, Bangjun Xiao, Bingquan Xia, Bofei Gao, Bowen Ye, Can Cai, Chang Liu, Chenhong He, Chunan Li, Dawei Zhu, Duo Zhang, Fengyuan Shi, Guoan Wang, Hailin Zhang, Hanglong Lv, Hanyu Li, Hao Tian, Heng Qu, Hongshen Xu, Houbin Zhang, Huaqiu Liu, Jiangshan Duo, Jianguang Zuo, Jianyu Wei, Jiebao Xiao, Jinhao Dong, Jun Shi, Junhao Hu, Kainan Bao, Kang Zhou, Linghao Zhang, Meng Chen, Nuo Chen, Peng Zhang, Qianli Chen, Qiantong Wang, Rang Li, Shaohui Liu, Shengfan Wang, Shicheng Li, Shihua Yu, Shijie Cao, Shimao Chen, Shuhao Gu, Weikun Wang, Wenhan Ma, Xiangwei Deng, Xing Yong, Xing Zhang, Xu Wang, Yifan Song, Yihao Zhao, Yingbo Zhao, Yizhao Gao, Yu Cheng, Yu Tu, Yudong Wang, Zhaojun Huang, Zhengju Tang, Zhenru Lin, Zhichao Song, Zhipeng Xu, Zhixian Zheng, Zihan Jiang

**发布时间:** 2025-12-29

### GPT解析

### 总结

论文介绍了MiMo-Audio模型，一个大规模预训练音频语言模型，通过扩展预训练数据到超过一亿小时实现了少样本学习能力，在多种音频任务上取得了最先进性能，并开源了模型和评估套件。

### 背景

现有音频语言模型通常依赖任务特定微调，而人类只需少量例子或简单指令就能泛化到新任务。GPT-3在文本领域展示了大规模预训练的泛化能力，这种范式同样适用于音频领域。

### 目的

探索大规模预训练是否能实现音频领域的少样本学习和泛化能力，开发一个能够理解和生成音频的通用模型。

### 方法

将MiMo-Audio预训练数据扩展到超过一亿小时，开发系统评估方法，在后训练阶段策划多样化指令微调语料库，并在音频理解和生成中引入思考机制。

### 主要发现

MiMo-Audio-7B-Base在开源模型中实现了语音智能和音频理解基准测试的最先进性能；能泛化到训练数据中不存在的任务如语音转换、风格迁移和语音编辑；展示强大语音续写能力，可生成逼真谈话节目、朗诵、直播和辩论；MiMo-Audio-7B-Instruct在多种基准测试中接近或超越闭源模型。

### 结论

大规模预训练范式同样适用于音频领域，MiMo-Audio模型展示了强大的少样本学习能力和泛化能力，在多种音频任务上取得了接近或超越闭源模型的最先进性能。

### 翻译

现有的音频语言模型通常依赖于任务特定的微调来完成特定的音频任务。相比之下，人类只需几个例子或简单指令就能泛化到新的音频任务。GPT-3已经证明，扩展下一个词预测预训练可以在文本中实现强大的泛化能力，我们认为这种范式同样适用于音频领域。通过将MiMo-Audio的预训练数据扩展到超过一亿小时，我们观察到在多样化的音频任务集上出现了少样本学习能力。我们开发了对这些能力的系统评估，发现MiMo-Audio-7B-Base在开源模型中同时实现了语音智能和音频理解基准测试的最先进性能。除了标准指标外，MiMo-Audio-7B-Base还能泛化到训练数据中不存在的任务，如语音转换、风格迁移和语音编辑。MiMo-Audio-7B-Base还展示了强大的语音续写能力，能够生成高度逼真的谈话节目、朗诵、直播和辩论。在后训练阶段，我们策划了多样化的指令微调语料库，并在音频理解和生成中引入了思考机制。MiMo-Audio-7B-Instruct在音频理解基准测试、口语对话基准测试和指令-TTS评估中达到了开源最先进水平，接近或超越闭源模型。模型检查点和完整评估套件可在指定网站获取。


### 论文摘要

Existing audio language models typically rely on task-specific fine-tuning to accomplish particular audio tasks. In contrast, humans are able to generalize to new audio tasks with only a few examples or simple instructions. GPT-3 has shown that scaling next-token prediction pretraining enables strong generalization capabilities in text, and we believe this paradigm is equally applicable to the audio domain. By scaling MiMo-Audio's pretraining data to over one hundred million of hours, we observe the emergence of few-shot learning capabilities across a diverse set of audio tasks. We develop a systematic evaluation of these capabilities and find that MiMo-Audio-7B-Base achieves SOTA performance on both speech intelligence and audio understanding benchmarks among open-source models. Beyond standard metrics, MiMo-Audio-7B-Base generalizes to tasks absent from its training data, such as voice conversion, style transfer, and speech editing. MiMo-Audio-7B-Base also demonstrates powerful speech continuation capabilities, capable of generating highly realistic talk shows, recitations, livestreaming and debates. At the post-training stage, we curate a diverse instruction-tuning corpus and introduce thinking mechanisms into both audio understanding and generation. MiMo-Audio-7B-Instruct achieves open-source SOTA on audio understanding benchmarks (MMSU, MMAU, MMAR, MMAU-Pro), spoken dialogue benchmarks (Big Bench Audio, MultiChallenge Audio) and instruct-TTS evaluations, approaching or surpassing closed-source models. Model checkpoints and full evaluation suite are available at https://github.com/XiaomiMiMo/MiMo-Audio.

---

## 63. End-to-End Test-Time Training for Long Context

**论文链接:** [http://arxiv.org/abs/2512.23675v2](http://arxiv.org/abs/2512.23675v2)

**作者:** Arnuv Tandon, Karan Dalal, Xinhao Li, Daniel Koceja, Marcel Rød, Sam Buchanan, Xiaolong Wang, Jure Leskovec, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin, Jed McCaleb, Yejin Choi, Yu Sun

**发布时间:** 2025-12-29

**备注:** Code: https://github.com/test-time-training/e2e

### GPT解析

### 总结

论文将长上下文语言建模重新表述为持续学习问题而非架构设计问题，提出了一种测试时训练(TTT-E2E)方法，在测试时通过下一个词预测持续学习，将上下文压缩到模型权重中。

### 背景

现有长上下文语言建模方法主要关注架构设计，而本文提出将问题视为持续学习问题。

### 目的

开发一种在测试时也能持续学习的端到端方法，解决长上下文语言建模问题。

### 方法

使用带有滑动窗口注意力的标准Transformer架构，在测试时通过下一个词预测进行持续学习，并通过训练时的元学习改进模型初始化。

### 主要发现

对于3B模型(使用164B tokens训练)，TTT-E2E与全注意力Transformer一样随上下文长度扩展；与Mamba 2和Gated DeltaNet不同；具有恒定的推理延迟，与上下文长度无关；对于128K上下文，比全注意力快2.7倍。

### 结论

TTT-E2E方法在长上下文语言建模中表现出色，代码已公开可用。

### 翻译

我们将长上下文语言建模表述为一个持续学习问题而非架构设计问题。在此表述下，我们仅使用标准架构——带有滑动窗口注意力的Transformer。然而，我们的模型通过在给定上下文上进行下一个词预测来在测试时持续学习，将其读取的上下文压缩到权重中。此外，我们通过训练时的元学习改进模型初始化，以便在测试时学习。总体而言，我们的方法——一种测试时训练(TTT)形式——在测试时(通过下一个词预测)和训练时(通过元学习)都是端到端的，这与之前的形式不同。我们进行了广泛实验，重点关注扩展特性。特别是，对于使用164B tokens训练的3B模型，我们的方法(TTT-E2E)与全注意力Transformer一样随上下文长度扩展，而其他方法如Mamba 2和Gated DeltaNet则不能。然而，类似于RNN，TTT-E2E具有恒定的推理延迟，与上下文长度无关，使其在128K上下文时比全注意力快2.7倍。我们的代码已公开可用。


### 论文摘要

We formulate long-context language modeling as a problem in continual learning rather than architecture design. Under this formulation, we only use a standard architecture -- a Transformer with sliding-window attention. However, our model continues learning at test time via next-token prediction on the given context, compressing the context it reads into its weights. In addition, we improve the model's initialization for learning at test time via meta-learning at training time. Overall, our method, a form of Test-Time Training (TTT), is End-to-End (E2E) both at test time (via next-token prediction) and training time (via meta-learning), in contrast to previous forms. We conduct extensive experiments with a focus on scaling properties. In particular, for 3B models trained with 164B tokens, our method (TTT-E2E) scales with context length in the same way as Transformer with full attention, while others, such as Mamba 2 and Gated DeltaNet, do not. However, similar to RNNs, TTT-E2E has constant inference latency regardless of context length, making it 2.7 times faster than full attention for 128K context. Our code is publicly available.

---

## 64. Coordinate Matrix Machine: A Human-level Concept Learning to Classify Very Similar Documents

**论文链接:** [http://arxiv.org/abs/2512.23749v1](http://arxiv.org/abs/2512.23749v1)

**作者:** Amin Sadri, M Maruf Hossain

**发布时间:** 2025-12-26

**备注:** 16 pages, 3 figures

### GPT解析

### 总结

本文提出Coordinate Matrix Machine (CM²)模型，这是一种小型模型，通过学习文档结构实现人类水平的概念学习能力，每个类别仅需一个样本即可分类文档，同时作为绿色AI解决方案减少对计算资源的依赖。

### 背景

人类水平概念学习表明人类通常只需一个示例就能学习新概念，而机器学习算法需要数百个样本。人类大脑能潜意识识别重要特征并高效学习，而现代'红色AI'趋势依赖大规模预训练和能源密集型GPU基础设施。

### 目的

开发实现人类水平概念学习的模型，通过识别结构性'重要特征'实现文档分类，每个类别只需一个样本，同时设计绿色AI解决方案减少资源消耗。

### 方法

提出Coordinate Matrix Machine (CM²)模型，一种专为增强人类智能设计的小型系统，通过学习文档结构并利用这些信息分类文档，专注于结构坐标而非详尽语义向量。

### 主要发现

CM²算法性能优于传统向量化器和复杂深度学习模型，提供八大优势：最小数据下高精度、几何和结构智能、绿色AI环保、CPU环境优化、内在可解释性、快速计算低延迟、对不平衡类的鲁棒性、经济可行性以及通用可扩展性。

### 结论

CM²模型成功实现人类水平概念学习能力，通过关注结构特征而非大量数据，提供高效、环保且可解释的AI解决方案，代表向更高效可持续AI系统的重要转变。

### 翻译

人类水平概念学习认为，人类通常从单个示例学习新概念，而机器学习算法通常需要数百个样本来学习单个概念。我们的大脑潜意识地识别重要特征并更有效地学习。贡献：在本文中，我们提出了坐标矩阵机(CM²)。这个专门设计的小型模型通过学习文档结构并利用这些信息来分类文档来增强人类智能。虽然现代'红色AI'趋势依赖于大规模预训练和能源密集型GPU基础设施，但CM²被设计为绿色AI解决方案。它通过识别人类会考虑的结构性'重要特征'，实现了人类水平的概念学习，使其能够仅使用每个类别的一个样本来分类非常相似的文档。优势：我们的算法性能优于传统的向量化器和需要更大数据集和大量计算的复杂深度学习模型。通过专注于结构坐标而非详尽的语义向量，CM²提供了：1. 最小数据下的高精度（一次性学习）2. 几何和结构智能 3. 绿色AI和环境可持续性 4. 仅针对CPU环境优化 5. 内在可解释性（玻璃盒模型）6. 更快的计算和低延迟 7. 对不平衡类的鲁棒性 8. 经济可行性 9. 通用、可扩展和可扩展


### 论文摘要

Human-level concept learning argues that humans typically learn new concepts from a single example, whereas machine learning algorithms typically require hundreds of samples to learn a single concept. Our brain subconsciously identifies important features and learns more effectively. \vspace*{6pt}   Contribution: In this paper, we present the Coordinate Matrix Machine (CM$^2$). This purpose-built small model augments human intelligence by learning document structures and using this information to classify documents. While modern "Red AI" trends rely on massive pre-training and energy-intensive GPU infrastructure, CM$^2$ is designed as a Green AI solution. It achieves human-level concept learning by identifying only the structural "important features" a human would consider, allowing it to classify very similar documents using only one sample per class.   Advantage: Our algorithm outperforms traditional vectorizers and complex deep learning models that require larger datasets and significant compute. By focusing on structural coordinates rather than exhaustive semantic vectors, CM$^2$ offers: 1. High accuracy with minimal data (one-shot learning) 2. Geometric and structural intelligence 3. Green AI and environmental sustainability 4. Optimized for CPU-only environments 5. Inherent explainability (glass-box model) 6. Faster computation and low latency 7. Robustness against unbalanced classes 8. Economic viability 9. Generic, expandable, and extendable

---

## 65. MoniRefer: A Real-world Large-scale Multi-modal Dataset based on Roadside Infrastructure for 3D Visual Grounding

**论文链接:** [http://arxiv.org/abs/2512.24605v1](http://arxiv.org/abs/2512.24605v1)

**作者:** Panquan Yang, Junfei Huang, Zongzhangbao Yin, Yingsong Hu, Anni Xu, Xinyi Luo, Xueqi Sun, Hai Wu, Sheng Ao, Zhaoxing Zhu, Chenglu Wen, Cheng Wang

**发布时间:** 2025-12-31

**备注:** 14 pages

### GPT解析

### 总结

该论文介绍了3D视觉定位在户外监控场景中的应用，提出了新的任务、数据集和方法

### 背景

3D视觉定位旨在定位3D点云场景中与给定自然语言语义对应的目标对象，这对路边基础设施系统在复杂交通环境中解释自然语言和定位目标至关重要。然而，现有方法主要关注室内和室外驾驶场景，户外监控场景因缺乏相关数据而未被充分探索

### 目的

引入3D视觉定位在户外监控场景的新任务，使基础设施能够超越自车视角理解交通场景，并为此构建相应的数据集

### 方法

提出名为Moni3DVG的新端到端方法，利用图像的外观信息和点云的几何及光学信息进行多模态特征学习和3D目标定位

### 主要发现

在提出的基准上进行的广泛实验和消融研究证明了Moni3DVG方法的优越性和有效性

### 结论

作者将公开发布MoniRefer数据集和Moni3DVG方法的代码

### 翻译

三维视觉定位旨在定位三维点云场景中与给定自然语言句子语义上相对应的目标对象。对于路边基础设施系统来说，在复杂的交通环境中解释自然语言并定位相关目标对象非常重要。然而，大多数现有的三维视觉定位数据集和方法都集中在室内和室外驾驶场景，由于路边基础设施传感器捕获的配对点云-文本数据稀缺，户外监控场景仍未被探索。在本文中，我们引入了户外监控场景的三维视觉定位这一新任务，使基础设施能够超越自车视角理解交通场景。为支持这一任务，我们构建了MoniRefer，这是首个用于路边级三维视觉定位的真实世界大规模多模态数据集。该数据集包含约136,018个目标和411,128条自然语言表达，这些数据从真实世界环境中的多个复杂交通路口收集。为确保数据集的质量和准确性，我们人工核对了所有目标的语言描述和三维标签。此外，我们还提出了一种名为Moni3DVG的新端到端方法，该方法利用图像提供的丰富外观信息和点云的几何及光学信息进行多模态特征学习和三维目标定位。在提出的基准上进行的广泛实验和消融研究证明了我们方法的优越性和有效性。我们将发布我们的数据集和代码。


### 论文摘要

3D visual grounding aims to localize the object in 3D point cloud scenes that semantically corresponds to given natural language sentences. It is very critical for roadside infrastructure system to interpret natural languages and localize relevant target objects in complex traffic environments. However, most existing datasets and approaches for 3D visual grounding focus on the indoor and outdoor driving scenes, outdoor monitoring scenarios remain unexplored due to scarcity of paired point cloud-text data captured by roadside infrastructure sensors. In this paper, we introduce a novel task of 3D Visual Grounding for Outdoor Monitoring Scenarios, which enables infrastructure-level understanding of traffic scenes beyond the ego-vehicle perspective. To support this task, we construct MoniRefer, the first real-world large-scale multi-modal dataset for roadside-level 3D visual grounding. The dataset consists of about 136,018 objects with 411,128 natural language expressions collected from multiple complex traffic intersections in the real-world environments. To ensure the quality and accuracy of the dataset, we manually verified all linguistic descriptions and 3D labels for objects. Additionally, we also propose a new end-to-end method, named Moni3DVG, which utilizes the rich appearance information provided by images and geometry and optical information from point cloud for multi-modal feature learning and 3D object localization. Extensive experiments and ablation studies on the proposed benchmarks demonstrate the superiority and effectiveness of our method. Our dataset and code will be released.

---

## 66. Geometric Multi-Session Map Merging with Learned Local Descriptors

**论文链接:** [http://arxiv.org/abs/2512.24384v1](http://arxiv.org/abs/2512.24384v1)

**作者:** Yanlong Ma, Nakul S. Joshi, Christa S. Robison, Philip R. Osteen, Brett T. Lopez

**发布时间:** 2025-12-30

### GPT解析

### 总结

本文提出了GMLD，一个基于学习的局部描述符框架，用于大规模多会话点云地图合并，能够系统地对齐不同会话中收集的具有重叠区域的地图。

### 背景

多会话地图合并对于大规模环境中的扩展自主操作至关重要。

### 目的

开发一个能够有效合并不同会话收集的点云地图的方法，确保地图的全局一致性。

### 方法

提出了一个关键点感知编码器和基于平面的几何变换器，提取用于循环闭合检测和相对姿态估计的判别性特征；在因子图优化阶段引入了会话间扫描匹配成本因子，以提高全局一致性。

### 主要发现

在公共数据集和不同环境中自收集的数据上评估，结果显示地图合并准确且鲁棒，误差低；学习到的特征在循环闭合检测和相对姿态估计中表现出色。

### 结论

GMLD框架能够实现准确且鲁棒的多会话点云地图合并，学习到的特征在关键任务中表现出强大性能。

### 翻译

多会话地图合并对于大规模环境中的扩展自主操作至关重要。在本文中，我们提出了GMLD，一个基于学习的局部描述符框架，用于大规模多会话点云地图合并，能够系统地对齐不同会话中收集的具有重叠区域的地图。所提出的框架采用关键点感知编码器和基于平面的几何变换器，提取用于循环闭合检测和相对姿态估计的判别性特征。为进一步提高全局一致性，我们在因子图优化阶段包含了会话间扫描匹配成本因子。我们在公共数据集以及从不同环境中自收集的数据上评估了我们的框架。结果表明，地图合并准确且鲁棒，误差低，并且学习到的特征在循环闭合检测和相对姿态估计中都表现出强大的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多会话地图融合的问题，即如何将不同时间或不同智能体收集的地图整合为一个全局一致的地图。这个问题在自动驾驶、基础设施检查和探索等自主应用中非常重要，因为它允许系统在大型环境中持续运行，而不受单次会话范围的限制，同时保持全局一致性，为自主系统提供可靠的环境理解基础。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有地图融合方法面临的挑战，包括数据稀疏性、传感器视角变化、高计算需求等。他们借鉴了现有的特征提取技术，如PointNetVLAD、MinkLoc3D等，但进行了改进。特别是，作者引入了一个关键点感知的下采样策略来解决传统下采样方法的不一致性问题，并设计了一个基于平面的几何变换器来增强特征提取。此外，作者还改进了因子图优化方法，增加了会话间扫描匹配成本因子来提高全局一致性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用学习到的局部描述符来识别不同会话地图中的重叠区域，并估计它们之间的相对变换，然后将这些地图整合到一个全局一致的坐标系中。整体实现流程包括三个主要模块：1) 关键点和描述符生成：从点云中提取一致的关键点和对应的局部描述符；2) 回路闭合检测和配准：通过计算描述符距离识别潜在重叠区域，并估计相对变换；3) 地图融合：构建包含相对姿态因子和扫描匹配成本因子的因子图，优化后得到全局一致的融合地图。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 关键点感知的下采样策略，提高了局部描述符的一致性；2) 基于平面的几何变换器，增强了关键点之间的内部几何关系；3) 会话间扫描匹配成本因子，在因子图优化中显式处理重叠区域的几何一致性；4) 完整的3D大规模地图融合流程。相比之前的工作，该方法能够同时处理显式和隐式的回路闭合，使用更精确的几何特征表示，并在因子图优化中考虑扫描匹配成本，从而显著提高了全局一致性和鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于学习局部描述符的几何多会话地图融合框架，通过关键点感知下采样、平面几何变换和扫描匹配成本因子优化，实现了在大规模环境中准确、鲁棒地融合来自不同会话的LiDAR点云地图。'}


### 论文摘要

Multi-session map merging is crucial for extended autonomous operations in large-scale environments. In this paper, we present GMLD, a learning-based local descriptor framework for large-scale multi-session point cloud map merging that systematically aligns maps collected across different sessions with overlapping regions. The proposed framework employs a keypoint-aware encoder and a plane-based geometric transformer to extract discriminative features for loop closure detection and relative pose estimation. To further improve global consistency, we include inter-session scan matching cost factors in the factor-graph optimization stage. We evaluate our framework on the public datasets, as well as self-collected data from diverse environments. The results show accurate and robust map merging with low error, and the learned features deliver strong performance in both loop closure detection and relative pose estimation.

---

## 67. Topological Spatial Graph Coarsening

**论文链接:** [http://arxiv.org/abs/2512.24327v1](http://arxiv.org/abs/2512.24327v1)

**作者:** Anna Calissano, Etienne Lasalle

**发布时间:** 2025-12-30

### GPT解析

### 总结

该研究提出了一种拓扑空间图粗化方法，通过折叠短边实现空间图约简，同时保持图的拓扑特征。该方法基于新的三角形感知图过滤框架，无参数且在几何变换下具有不变性，能有效减少图大小并保留重要拓扑信息。

### 背景

空间图是一类特殊图，其节点在空间中有特定位置，如公共交通网络、分子结构和分支生物结构。这类图在保留空间信息的同时进行约简具有挑战性。

### 目的

研究旨在开发一种空间图约简方法，能够在减少节点数量的同时保持初始图的拓扑特征和整体结构。

### 方法

提出一种拓扑空间图粗化方法，通过折叠短边实现约简。该方法引入了三角形感知图过滤框架，将经典拓扑描述符（持续图）从点云适应到空间图，以捕捉校准约简级别所需的拓扑信息。

### 主要发现

所提出的粗化方法是无参数的，在初始空间图的旋转、平移和缩放下保持不变。实验表明该方法能显著减少图的大小，同时保留相关拓扑信息。

### 结论

该拓扑空间图粗化方法在图约简和拓扑特征保持之间取得了有效平衡，适用于合成和真实空间图，是一种高效的空间图简化技术。

### 翻译

空间图是一类特殊图，其节点在空间中有位置（例如公共交通网络、分子、分支生物结构）。在这项工作中，我们考虑空间图约简问题，旨在找到一个更小的空间图（即节点更少）但与初始图具有相同整体结构。在此背景下，由于额外的空间信息，执行图约简同时保持初始图的主要拓扑特征尤为重要。因此，我们提出了一种基于新框架的拓扑空间图粗化方法，该方法在图约简和拓扑特征保持之间找到平衡。粗化通过折叠短边实现。为了捕捉校准约简级别所需的拓扑信息，我们将为点云构建的经典拓扑描述符（所谓的持续图）适应到空间图。这种构造依赖于引入一种称为三角形感知图过滤的新过滤方法。我们的粗化方法是无参数的，我们证明了它在初始空间图的旋转、平移和缩放下具有等变性。我们在合成和真实空间图上评估了我们方法的性能，表明它在显著减少图大小的同时保留了相关拓扑信息。


### 论文摘要

Spatial graphs are particular graphs for which the nodes are localized in space (e.g., public transport network, molecules, branching biological structures). In this work, we consider the problem of spatial graph reduction, that aims to find a smaller spatial graph (i.e., with less nodes) with the same overall structure as the initial one. In this context, performing the graph reduction while preserving the main topological features of the initial graph is particularly relevant, due to the additional spatial information. Thus, we propose a topological spatial graph coarsening approach based on a new framework that finds a trade-off between the graph reduction and the preservation of the topological characteristics. The coarsening is realized by collapsing short edges. In order to capture the topological information required to calibrate the reduction level, we adapt the construction of classical topological descriptors made for point clouds (the so-called persistent diagrams) to spatial graphs. This construction relies on the introduction of a new filtration called triangle-aware graph filtration. Our coarsening approach is parameter-free and we prove that it is equivariant under rotations, translations and scaling of the initial spatial graph. We evaluate the performances of our method on synthetic and real spatial graphs, and show that it significantly reduces the graph sizes while preserving the relevant topological information.

---

## 68. RANGER: A Monocular Zero-Shot Semantic Navigation Framework through Contextual Adaptation

**论文链接:** [http://arxiv.org/abs/2512.24212v1](http://arxiv.org/abs/2512.24212v1)

**作者:** Ming-Ming Yu, Yi Chen, Börje F. Karlsson, Wenjun Wu

**发布时间:** 2025-12-30

### GPT解析

### 总结

RANGER 是一个零样本、开放词汇表的语义导航框架，仅使用单目相机操作，消除了对深度和姿态信息的依赖，并具备上下文学习能力。

### 背景

在复杂环境中高效寻找目标是现实世界应用的基础。虽然最近的多模态基础模型已经实现了零样本目标导航，但现有方法面临两个主要限制：严重依赖模拟器提供的精确深度和姿态信息，限制了现实应用；缺乏上下文学习能力，难以快速适应新环境。

### 目的

解决现有方法的两个关键限制，提出一个不依赖深度和姿态信息且具备上下文学习能力的语义导航框架。

### 方法

提出了 RANGER 框架，利用强大的 3D 基础模型消除对深度和姿态的依赖，同时展现强大的上下文学习能力。通过简单观察新环境的短视频，系统可以显著提高任务效率，无需架构修改或微调。框架集成了基于关键帧的 3D 重建、语义点云生成、视觉语言模型驱动的探索价值估计、高层自适应路径点选择和底层动作执行等关键组件。

### 主要发现

在 HM3D 基准测试和真实环境中的实验表明，RANGER 在导航成功率和探索效率方面实现了具有竞争力的性能，同时表现出优越的上下文学习适应性，且无需预先对环境进行 3D 映射。

### 结论

RANGER 框架解决了现有方法在现实世界应用中的两个关键限制，为语义导航提供了新的解决方案，无需深度和姿态信息即可实现高效的零样本目标导航。

### 翻译

在复杂环境中高效寻找目标是现实世界具身应用的基础。虽然最近的多模态基础模型的进展已经实现了零样本目标导航，允许机器人无需微调即可搜索任意物体，但现有方法面临两个关键限制：(1) 严重依赖模拟器提供的精确深度和姿态信息，这限制了在现实场景中的适用性；(2) 缺乏上下文学习能力，使得难以快速适应新环境，例如利用短视频。为解决这些挑战，我们提出了 RANGER，一个新颖的零样本、开放词汇表的语义导航框架，仅使用单目相机运行。利用强大的 3D 基础模型，RANGER 消除了对深度和姿态的依赖，同时展现出强大的上下文学习能力。通过简单观察新环境的短视频，系统也可以显著提高任务效率，无需架构修改或微调。该框架集成了几个关键组件：基于关键帧的 3D 重建、语义点云生成、视觉语言模型驱动的探索价值估计、高层自适应路径点选择和底层动作执行。在 HM3D 基准测试和真实环境中的实验表明，RANGER 在导航成功率和探索效率方面实现了具有竞争力的性能，同时表现出优越的上下文学习适应性，无需预先对环境进行 3D 映射。


### 论文摘要

Efficiently finding targets in complex environments is fundamental to real-world embodied applications. While recent advances in multimodal foundation models have enabled zero-shot object goal navigation, allowing robots to search for arbitrary objects without fine-tuning, existing methods face two key limitations: (1) heavy reliance on precise depth and pose information provided by simulators, which restricts applicability in real-world scenarios; and (2) lack of in-context learning (ICL) capability, making it difficult to quickly adapt to new environments, as in leveraging short videos. To address these challenges, we propose RANGER, a novel zero-shot, open-vocabulary semantic navigation framework that operates using only a monocular camera. Leveraging powerful 3D foundation models, RANGER eliminates the dependency on depth and pose while exhibiting strong ICL capability. By simply observing a short video of a new environment, the system can also significantly improve task efficiency without requiring architectural modifications or fine-tuning. The framework integrates several key components: keyframe-based 3D reconstruction, semantic point cloud generation, vision-language model (VLM)-driven exploration value estimation, high-level adaptive waypoint selection, and low-level action execution. Experiments on the HM3D benchmark and real-world environments demonstrate that RANGER achieves competitive performance in terms of navigation success rate and exploration efficiency, while showing superior ICL adaptability, with no previous 3D mapping of the environment required.

---

## 69. PointRAFT: 3D deep learning for high-throughput prediction of potato tuber weight from partial point clouds

**论文链接:** [http://arxiv.org/abs/2512.24193v1](http://arxiv.org/abs/2512.24193v1)

**作者:** Pieter M. Blok, Haozhou Wang, Hyun Kwon Suh, Peicheng Wang, James Burridge, Wei Guo

**发布时间:** 2025-12-30

**备注:** 14 pages, 7 figures, 3 tables

### GPT解析

### 总结

本文介绍了一种名为PointRAFT的高通量点云回归网络，用于从RGB-D相机捕获的部分点云直接预测马铃薯重量，解决了因自遮挡导致的点云不完整问题，实现了精确且高效的马铃薯重量估计。

### 背景

马铃薯产量是优化农业栽培实践的关键指标。使用RGB-D相机可在收获机上估计马铃薯产量，捕捉传送带上移动块茎的三维信息。然而，从RGB-D图像重建的点云因自遮挡而不完整，导致对块茎重量的系统性低估。

### 目的

开发一种能够从部分点云直接预测连续3D形状属性（如块茎重量）的高通量点云回归网络，以解决点云不完整导致的重量低估问题。

### 方法

提出了PointRAFT网络，不重建完整3D几何形状，而是直接从原始3D数据推断目标值。关键创新是对象高度嵌入，将块茎高度作为额外几何线索纳入。使用从日本收获机上收集的4个品种和3个生长季节的859个马铃薯块茎上的26,688个部分点云进行训练和评估。

### 主要发现

在测试集上（172个块茎的5,254个点云），PointRAFT平均绝对误差为12.0克，均方根误差为17.2克，显著优于线性回归基线和PointNet++网络。每个点云平均推理时间为6.3毫秒，支持每秒处理150个块茎，满足商业收获机的高通量要求。

### 结论

PointRAFT不仅适用于马铃薯重量估计，还提供了适用于各种3D表型和机器人感知任务的通用回归网络。代码、网络权重和数据集子集已在GitHub上公开。

### 翻译

马铃薯产量是优化农业栽培实践的关键指标。可以在收获机上使用RGB-D相机估计马铃薯产量，这些相机能够捕捉沿着传送带移动的个体块茎的三维信息。然而，从RGB-D图像重建的点云由于自遮挡而不完整，导致对块茎重量的系统性低估。为此，我们引入了PointRAFT，一种高通量点云回归网络，可以直接从部分点云预测连续的3D形状属性，如块茎重量。PointRAFT不重建完整的3D几何形状，而是直接从原始3D数据推断目标值。其关键架构创新是一种对象高度嵌入，将块茎高度作为额外的几何线索纳入，改善了实际收获条件下的重量预测。PointRAFT使用在日本一台运行中的收获机上收集的4个品种和3个生长季节的859个马铃薯块茎上的26,688个部分点云进行训练和评估。在包含172个块茎的5,254个点云的测试集上，PointRAFT实现了12.0克的平均绝对误差和17.2克的均方根误差，显著优于线性回归基线和标准PointNet++回归网络。每个点云的平均推理时间为6.3毫秒，支持每秒最多处理150个块茎的速度，满足了商业马铃薯收获机的高通量要求。除了马铃薯重量估计外，PointRAFT还提供了一个适用于各种3D表型和机器人感知任务的通用回归网络。代码、网络权重和数据集子集可在https://github.com/pieterblok/pointraft.git公开获取。


### 论文摘要

Potato yield is a key indicator for optimizing cultivation practices in agriculture. Potato yield can be estimated on harvesters using RGB-D cameras, which capture three-dimensional (3D) information of individual tubers moving along the conveyor belt. However, point clouds reconstructed from RGB-D images are incomplete due to self-occlusion, leading to systematic underestimation of tuber weight. To address this, we introduce PointRAFT, a high-throughput point cloud regression network that directly predicts continuous 3D shape properties, such as tuber weight, from partial point clouds. Rather than reconstructing full 3D geometry, PointRAFT infers target values directly from raw 3D data. Its key architectural novelty is an object height embedding that incorporates tuber height as an additional geometric cue, improving weight prediction under practical harvesting conditions. PointRAFT was trained and evaluated on 26,688 partial point clouds collected from 859 potato tubers across four cultivars and three growing seasons on an operational harvester in Japan. On a test set of 5,254 point clouds from 172 tubers, PointRAFT achieved a mean absolute error of 12.0 g and a root mean squared error of 17.2 g, substantially outperforming a linear regression baseline and a standard PointNet++ regression network. With an average inference time of 6.3 ms per point cloud, PointRAFT supports processing rates of up to 150 tubers per second, meeting the high-throughput requirements of commercial potato harvesters. Beyond potato weight estimation, PointRAFT provides a versatile regression network applicable to a wide range of 3D phenotyping and robotic perception tasks. The code, network weights, and a subset of the dataset are publicly available at https://github.com/pieterblok/pointraft.git.

---

## 70. DriveExplorer: Images-Only Decoupled 4D Reconstruction with Progressive Restoration for Driving View Extrapolation

**论文链接:** [http://arxiv.org/abs/2512.23983v1](http://arxiv.org/abs/2512.23983v1)

**作者:** Yuang Jia, Jinlong Wang, Jiayi Zhao, Chunlam Li, Shunzhou Wang, Wei Gao

**发布时间:** 2025-12-30

### GPT解析

### 总结

本文提出了一种仅使用图像和可选相机姿态的视点外推解决方案，通过结合点云估计、4D高斯重建和扩散模型迭代优化，实现了高质量的新视角图像生成。

### 背景

现有基于扩散模型的视点外推方法严重依赖LiDAR点云、3D边界框和车道标注等昂贵或耗时的先验信息，限制了其在现实世界部署中的适用性。

### 目的

开发一种无需昂贵传感器或复杂标注的视点外推方法，仅使用图像和可选相机姿态即可生成高质量的新视角图像。

### 方法

首先估计全局静态点云和每帧动态点云并融合为统一表示；然后采用可变形4D高斯框架重建场景；利用初始4D高斯模型渲染的图像训练视频扩散模型；通过扩散模型迭代优化渐进位移的高斯渲染，并将优化结果反馈回4DGS进行进一步训练，直至达到目标视点。

### 主要发现

与基线方法相比，该方法在新的外推视点上能够产生更高质量的图像，且不需要昂贵的传感器或复杂的标注工作。

### 结论

所提出的方法为自动驾驶场景中的视点外推提供了一种有效且实用的解决方案，克服了现有方法对先验信息的过度依赖问题。

### 翻译

本文提出了一种自动驾驶场景中视点外推的有效解决方案。近期的方法专注于使用扩散模型从给定视点生成位移的新视角图像。然而，这些方法严重依赖于LiDAR点云、3D边界框和车道标注等先验信息，这些信息需要昂贵的传感器或耗时的标注工作，限制了在现实世界部署中的适用性。在这项工作中，仅使用图像和可选的相机姿态，我们首先估计全局静态点云和每帧动态点云，将它们融合为统一表示。然后，我们采用可变形4D高斯框架来重建场景。初始训练的4D高斯模型渲染退化图像和伪图像，用于训练视频扩散模型。随后，渐进位移的高斯渲染通过扩散模型进行迭代细化，增强的结果被整合回4DGS的训练数据中。这个过程持续进行，直到外推达到目标视点。与基线方法相比，我们的方法在新的外推视点上产生更高质量的图像。


### 论文摘要

This paper presents an effective solution for view extrapolation in autonomous driving scenarios. Recent approaches focus on generating shifted novel view images from given viewpoints using diffusion models. However, these methods heavily rely on priors such as LiDAR point clouds, 3D bounding boxes, and lane annotations, which demand expensive sensors or labor-intensive labeling, limiting applicability in real-world deployment. In this work, with only images and optional camera poses, we first estimate a global static point cloud and per-frame dynamic point clouds, fusing them into a unified representation. We then employ a deformable 4D Gaussian framework to reconstruct the scene. The initially trained 4D Gaussian model renders degraded and pseudo-images to train a video diffusion model. Subsequently, progressively shifted Gaussian renderings are iteratively refined by the diffusion model,and the enhanced results are incorporated back as training data for 4DGS. This process continues until extrapolation reaches the target viewpoints. Compared with baselines, our method produces higher-quality images at novel extrapolated viewpoints.

---

## 71. SHIELD: Spherical-Projection Hybrid-Frontier Integration for Efficient LiDAR-based Drone Exploration

**论文链接:** [http://arxiv.org/abs/2512.23972v1](http://arxiv.org/abs/2512.23972v1)

**作者:** Liangtao Feng, Zhenchang Liu, Feng Zhang, Xuefeng Ren

**发布时间:** 2025-12-30

### GPT解析

### 总结

本文介绍了SHIELD，一种基于激光雷达的无人机高效探索方法，通过球面投影混合前沿集成解决了激光雷达在无人机探索中面临的挑战。

### 背景

激光雷达具有视场角宽的优势，但在无人机探索应用中面临三大挑战：激光雷达点云观测质量通常不如深度相机；传统前沿方法处理激光雷达宽视场角时计算负担重；无点云区域难以通过射线投射分类为自由空间。

### 目的

为了解决激光雷达在无人机探索中的质量问题、计算负担和区域分类问题，提高探索效率并确保飞行安全。

### 方法

SHIELD方法包括：维护观测质量占用地图并执行射线投射解决点云质量不一致问题；采用混合前沿方法处理计算负担和点云质量限制；提出向外球面投射射线策略在开放区域确保飞行安全和探索效率。

### 主要发现

通过模拟和飞行实验证明了SHIELD方法的有效性。

### 结论

SHIELD成功解决了激光雷达在无人机探索中的多项挑战，该方法将开源以促进研究社区的发展。

### 翻译

本文介绍了SHIELD，一种基于激光雷达的无人机高效探索的球面投影混合前沿集成方法。尽管激光雷达具有视场角宽的优势，但其在无人机探索中的应用仍面临几个挑战。激光雷达点云的观测质量通常不如深度相机。基于已知和未知区域的传统前沿方法施加了沉重的计算负担，特别是在处理激光雷达的宽视场角时。此外，没有点云的区域也难以通过射线投射分类为自由空间。为了解决这些问题，提出了SHIELD。它维护一个观测质量占用地图，并在该地图上进行射线投射，以解决探索过程中点云质量不一致的问题。使用混合前沿方法来处理计算负担和点云质量限制的探索。此外，提出了一种向外球面投射射线策略，在开放区域共同确保飞行安全和探索效率。模拟和飞行实验证明了SHIELD的有效性。这项工作将开源，为研究社区做出贡献。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决基于LiDAR的无人机自主探索中的三个关键问题：点云观测质量不均、传统前沿方法计算负担重、以及没有点云区域难以判断为自由空间。这些问题很重要，因为无人机在未知环境中的自主探索对灾害救援、地形测绘和建筑物检查等应用至关重要，而LiDAR相比深度相机有更宽的视场角，更适合大范围探索，提高探索效率可以节省时间和能源，准确的环境感知也能显著提高飞行安全性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有方法的局限性，指出传统方法在大视场角下会出现错误的射线投射，且大多数前沿位于地面降低了探索效率。作者借鉴了EPIC中使用观测距离构建占据地图的方法、基于前沿的探索策略以及多线程机制来平衡任务负载。在此基础上，作者设计了SHIELD方法，结合了质量感知的占据地图、混合前沿策略和外向球形投射射线投射方法，以解决LiDAR探索中的特定挑战。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建基于观测质量的占据地图解决射线投射错误问题，使用混合前沿策略平衡计算负担和探索效率，以及通过外向球形投射标记无点云区域的自由空间。整体流程包括：1)质量感知观测地图构建与射线投射，合并多周期点云并估计表面法线；2)混合前沿策略，检测并聚类质量前沿和未知前沿；3)外向球形投射射线投射，将点云投影到虚拟球面并进行自校准；4)路径规划，使用Hgrid划分空间并解决非对称旅行商问题确定访问顺序，最后进行局部路径规划。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点有三个：1)基于观测质量的占据地图和射线投射方法，考虑表面法线与射线方向的夹角作为质量指标；2)混合前沿策略，结合质量前沿和未知前沿；3)外向球形投射射线投射策略，解决无点云区域的自由空间标记问题。相比之前的工作特别是EPIC，SHIELD更好地处理了大视场角下的射线投射问题，混合前沿策略提供了更高的探索效率，外向球形投射使无人机能在开阔安全区域飞行，实验表明其在多种场景中表现出更快的探索速度和更短的路径长度。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SHIELD通过质量感知的占据地图、混合前沿策略和外向球形投射射线投射方法，显著提高了基于LiDAR的无人机在复杂环境中的自主探索效率和安全性。'}


### 论文摘要

This paper introduces SHIELD, a Spherical-Projection Hybrid-Frontier Integration for Efficient LiDAR-based Drone exploration method. Although laser LiDAR offers the advantage of a wide field of view, its application in UAV exploration still faces several challenges. The observation quality of LiDAR point clouds is generally inferior to that of depth cameras. Traditional frontier methods based on known and unknown regions impose a heavy computational burden, especially when handling the wide field of view of LiDAR. In addition, regions without point cloud are also difficult to classify as free space through raycasting. To address these problems, the SHIELD is proposed. It maintains an observation-quality occupancy map and performs ray-casting on this map to address the issue of inconsistent point-cloud quality during exploration. A hybrid frontier method is used to tackle both the computational burden and the limitations of point-cloud quality exploration. In addition, an outward spherical-projection ray-casting strategy is proposed to jointly ensure flight safety and exploration efficiency in open areas. Simulations and flight experiments prove the effectiveness of SHIELD. This work will be open-sourced to contribute to the research community.

---

## 72. 论文ID: 2512.23054v2

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23054v2.json'

---

## 73. 论文ID: 2512.22439v2

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22439v2.json'

---

## 74. 论文ID: 2512.24880v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.24880v1.json'

---

## 75. Characterization of Transfer Using Multi-task Learning Curves

**论文链接:** [http://arxiv.org/abs/2512.24866v1](http://arxiv.org/abs/2512.24866v1)

**作者:** András Millinghoffer, Bence Bolgár, Péter Antal

**发布时间:** 2025-12-31

### GPT解析

### 总结

本研究探讨了迁移效应的表征方法，提出通过扰动数据集而非模型来更基本地捕捉迁移效应，并使用多任务学习曲线进行定量建模。

### 背景

迁移效应在固定数据集训练和累积数据归纳推理中均有表现。现有研究主要通过梯度更新扰动模型来研究迁移效应。

### 目的

提出一种更基本的方法来表征迁移效应，通过增加样本扰动数据集而非扰动模型，并使用多任务学习曲线定量建模迁移效应。

### 方法

使用多任务学习曲线近似不同样本量下的归纳性能，描述了一种高效的方法来近似多任务学习曲线，类似于训练期间应用的任务亲和性分组方法。在基准药物-靶点相互作用数据集上进行了评估。

### 主要发现

学习曲线可以更好地捕捉多任务学习的效果，多任务扩展可以区分基础模型中的成对和上下文迁移效应。统计方法相比计算方法具有更高的计算成本，但更好的能力和更广泛的应用性。

### 结论

通过数据集扰动而非模型扰动来研究迁移效应，结合多任务学习曲线，能够更有效地捕捉和表征迁移效应，特别是在基础模型中的应用。

### 翻译

迁移效应在使用固定数据集的训练过程中和使用累积数据的归纳推理中都有体现。我们假设，通过包含更多样本来扰动数据集，而不是通过梯度更新来扰动模型，可以为迁移效应提供一种互补且更基本的表征。为了捕捉这种现象，我们使用多任务学习曲线定量建模迁移效应，这些曲线近似于不同样本量下的归纳性能。我们描述了一种高效的方法来近似多任务学习曲线，类似于训练期间应用的任务亲和性分组方法。我们比较了迁移的统计和计算方法，表明前者的计算成本要高得多，但具有更好的能力和更广泛的应用性。评估使用了一个基准药物-靶点相互作用数据集进行。我们的结果表明，学习曲线可以更好地捕捉多任务学习的效果，而它们的多任务扩展可以区分基础模型中的成对和上下文迁移效应。


### 论文摘要

Transfer effects manifest themselves both during training using a fixed data set and in inductive inference using accumulating data. We hypothesize that perturbing the data set by including more samples, instead of perturbing the model by gradient updates, provides a complementary and more fundamental characterization of transfer effects. To capture this phenomenon, we quantitatively model transfer effects using multi-task learning curves approximating the inductive performance over varying sample sizes. We describe an efficient method to approximate multi-task learning curves analogous to the Task Affinity Grouping method applied during training. We compare the statistical and computational approaches to transfer, which indicates considerably higher compute costs for the previous but better power and broader applicability. Evaluations are performed using a benchmark drug-target interaction data set. Our results show that learning curves can better capture the effects of multi-task learning and their multi-task extensions can delineate pairwise and contextual transfer effects in foundation models.

---

## 76. 论文ID: 2512.24849v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.24849v1.json'

---

## 77. 论文ID: 2512.24834v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.24834v1.json'

---

## 78. BandiK: Efficient Multi-Task Decomposition Using a Multi-Bandit Framework

**论文链接:** [http://arxiv.org/abs/2512.24708v1](http://arxiv.org/abs/2512.24708v1)

**作者:** András Millinghoffer, András Formanek, András Antos, Péter Antal

**发布时间:** 2025-12-31

**备注:** 8 pages, 14 figures

### GPT解析

### 总结

BandiK是一种新颖的三阶段多任务辅助任务子集选择方法，使用多老虎机框架解决多任务学习中辅助任务集选择面临的高计算成本、大量候选集和选择复杂性差异等问题。

### 背景

知识跨任务有效转移在基础模型的下游任务中至关重要，但转移的性质（传递性-非传递性）仍是一个开放问题，负迁移是显著障碍。多任务学习中辅助任务集的选择常受高计算成本、大量候选集和跨目标任务选择复杂性差异的限制。

### 目的

解决多任务学习中辅助任务集选择面临的高计算成本、大量合理的候选辅助集以及跨目标任务选择复杂性差异等问题。

### 方法

BandiK采用三阶段方法：1)估计任务间成对转移，识别可能从联合学习中受益的任务；2)基于初始估计为每个目标任务构建线性数量的候选辅助任务集；3)使用多臂老虎机框架评估候选辅助集性能。为提高效率，BandiK将单个任务特定的MAB整合到多老虎机结构中，利用相同神经网络实现不同老虎机的多个臂这一半重叠臂属性。

### 主要发现

负迁移是多任务学习中的一个重大障碍，需要有效的辅助任务选择方法来克服。

### 结论

BandiK通过三阶段多老虎机方法有效解决了多任务学习中辅助任务集选择的挑战，提高了效率和准确性。

### 翻译

在多个任务间有效转移知识的挑战至关重要，并且在基础模型的下游任务中也存在。然而，转移的性质及其传递性-非传递性仍是一个开放问题，负迁移仍然是一个重大障碍。多任务学习中辅助任务集的选择经常受到评估的高计算成本、大量合理的候选辅助集以及跨目标任务选择复杂性差异的限制。为解决这些限制，我们引入BandiK，一种新颖的三阶段多任务辅助任务子集选择方法，使用多老虎机，其中每次拉杆评估候选辅助集，通过在单个随机训练-测试数据集分割上训练和测试多输出神经网络来实现。首先，BandiK估计任务间的成对转移，帮助识别哪些任务可能从联合学习中受益。在第二阶段，它基于初始估计为每个目标任务构建线性数量的候选辅助任务集（任务数量的线性），显著减少了潜在的辅助任务集数量。第三，它为每个任务采用多臂老虎机框架，其中臂对应于候选辅助集在训练-测试数据集分割上的性能，实现为多输出神经网络。为提高效率，BandiK将这些单个任务特定的MAB整合到多老虎机结构中。所提出的多老虎机解决方案利用了相同的神经网络实现了不同单个老虎机的多个臂，对应于给定的候选集。这种半重叠臂属性定义了BandiK中使用的新型多老虎机成本/奖励结构。


### 论文摘要

The challenge of effectively transferring knowledge across multiple tasks is of critical importance and is also present in downstream tasks with foundation models. However, the nature of transfer, its transitive-intransitive nature, is still an open problem, and negative transfer remains a significant obstacle. Selection of beneficial auxiliary task sets in multi-task learning is frequently hindered by the high computational cost of their evaluation, the high number of plausible candidate auxiliary sets, and the varying complexity of selection across target tasks.   To address these constraints, we introduce BandiK, a novel three-stage multi-task auxiliary task subset selection method using multi-bandits, where each arm pull evaluates candidate auxiliary sets by training and testing a multiple output neural network on a single random train-test dataset split. Firstly, BandiK estimates the pairwise transfers between tasks, which helps in identifying which tasks are likely to benefit from joint learning. In the second stage, it constructs a linear number of candidate sets of auxiliary tasks (in the number of all tasks) for each target task based on the initial estimations, significantly reducing the exponential number of potential auxiliary task sets. Thirdly, it employs a Multi-Armed Bandit (MAB) framework for each task, where the arms correspond to the performance of candidate auxiliary sets realized as multiple output neural networks over train-test data set splits. To enhance efficiency, BandiK integrates these individual task-specific MABs into a multi-bandit structure. The proposed multi-bandit solution exploits that the same neural network realizes multiple arms of different individual bandits corresponding to a given candidate set. This semi-overlapping arm property defines a novel multi-bandit cost/reward structure utilized in BandiK.

---

## 79. 论文ID: 2512.24511v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.24511v1.json'

---

## 80. F2IDiff: Real-world Image Super-resolution using Feature to Image Diffusion Foundation Model

**论文链接:** [http://arxiv.org/abs/2512.24473v1](http://arxiv.org/abs/2512.24473v1)

**作者:** Devendra K. Jangid, Ripon K. Saha, Dilshan Godaliyadda, Jing Li, Seok-Jun Lee, Hamid R. Sheikh

**发布时间:** 2025-12-30

### GPT解析

### 总结

该研究提出了一种基于DINOv2特征的特征到图像扩散(F2IDiff)基础模型，解决了现有文本到图像扩散模型在智能手机超分辨率应用中的局限性。

### 背景

生成式AI显著提高了单图像超分辨率质量，但旗舰智能手机相机采用速度慢，因为强生成可能导致不希望的幻觉。在消费摄影中，LR图像保真度高，需要几乎没有幻觉的生成。

### 目的

解决文本特征无法准确描述小图像块纹理的问题，以及现有模型设计用于小图像(<1MP)而智能手机LR图像至少为12MP的尺寸不匹配问题。

### 方法

引入一个基于具有更低级别特征条件的基础模型的SISR网络，特别是DINOv2特征，称为特征到图像扩散(F2IDiff)基础模型(FM)。

### 主要发现

更低级别的特征提供了更严格的条件，同时是小的图像块的丰富描述符，能够更好地控制生成过程并减少不希望的幻觉。

### 结论

通过使用更低级别的特征条件，F2IDiff模型能够在保持生成质量的同时，更好地适应智能手机超分辨率应用的需求。

### 翻译

随着生成式AI的出现，单图像超分辨率(SISR)质量已显著提高，因为文本到图像扩散(T2IDiff)基础模型(FM)学习到的强先验能够弥合高分辨率(HR)和低分辨率(LR)图像之间的差距。然而，旗舰智能手机相机采用生成式模型的速度很慢，因为强生成可能导致不希望的幻觉。对于学术研究中看到的严重退化的LR图像，需要强生成，并且由于LR和HR图像之间的差距很大，幻觉更容易被容忍。相比之下，在消费摄影中，LR图像具有更高的保真度，只需要几乎没有幻觉的生成。我们假设SISR中的生成由基础模型的条件特征的严格性和丰富性控制。首先，文本特征是高级特征，通常无法描述图像中的微妙纹理。此外，智能手机LR图像至少为12MP，而基于T2IDiff FM构建的SISR网络设计用于在更小的图像(<1MP)上进行推理。因此，SISR推理必须在小的图像块上进行，而这些图像块通常无法被文本特征准确描述。为了解决这些缺点，我们引入了一个基于具有更低级别特征条件的基础模型的SISR网络，特别是DINOv2特征，我们称之为特征到图像扩散(F2IDiff)基础模型(FM)。更低级别的特征提供了更严格的条件，同时是小的图像块的丰富描述符。


### 论文摘要

With the advent of Generative AI, Single Image Super-Resolution (SISR) quality has seen substantial improvement, as the strong priors learned by Text-2-Image Diffusion (T2IDiff) Foundation Models (FM) can bridge the gap between High-Resolution (HR) and Low-Resolution (LR) images. However, flagship smartphone cameras have been slow to adopt generative models because strong generation can lead to undesirable hallucinations. For substantially degraded LR images, as seen in academia, strong generation is required and hallucinations are more tolerable because of the wide gap between LR and HR images. In contrast, in consumer photography, the LR image has substantially higher fidelity, requiring only minimal hallucination-free generation. We hypothesize that generation in SISR is controlled by the stringency and richness of the FM's conditioning feature. First, text features are high level features, which often cannot describe subtle textures in an image. Additionally, Smartphone LR images are at least $12MP$, whereas SISR networks built on T2IDiff FM are designed to perform inference on much smaller images ($<1MP$). As a result, SISR inference has to be performed on small patches, which often cannot be accurately described by text feature. To address these shortcomings, we introduce an SISR network built on a FM with lower-level feature conditioning, specifically DINOv2 features, which we call a Feature-to-Image Diffusion (F2IDiff) Foundation Model (FM). Lower level features provide stricter conditioning while being rich descriptors of even small patches.

---

## 81. OmniCosmos: Transferring Particle Physics Knowledge Across the Cosmos

**论文链接:** [http://arxiv.org/abs/2512.24422v1](http://arxiv.org/abs/2512.24422v1)

**作者:** Vinicius Mikuni, Ibrahim Elsharkawy, Benjamin Nachman

**发布时间:** 2025-12-30

**备注:** 7 pages, 5 figures

### GPT解析

### 总结

本研究展示了在对撞机数据上训练的基础模型可以跨科学领域泛化，应用于宇宙学参数预测和星系速度预测，这是粒子对撞机物理模型首次被证明能够跨科学领域应用。

### 背景

基础模型能够构建有效的数据表示，可用于多样化的下游任务。之前的研究开发了用于粒子对撞机物理学的OmniLearned基础模型，并展示了其在提高对撞机实验发现潜力方面的显著效果。

### 目的

本研究旨在探索在对撞机数据上训练的基础模型是否可以应用于粒子对撞机物理学以外的领域，特别是宇宙学研究，以改进宇宙学参数预测和星系速度预测。

### 方法

作者将在粒子对撞机数据上训练的基础模型应用于CosmoBench的不同数据集，用于宇宙学参数预测和晕及星系速度预测，评估其在跨科学领域任务中的性能。

### 主要发现

基础模型训练在对撞机数据上可以帮助改进宇宙学参数的预测，并能够预测不同CosmoBench数据集中的晕和星系速度，证明了粒子对撞机物理模型可以跨科学领域泛化。

### 结论

基础模型不仅在粒子对撞机物理学领域有效，还可以跨领域应用于宇宙学研究，这为科学模型在不同学科间的迁移学习提供了新思路。

### 翻译

基础模型构建了能够部署在多样化下游任务中的有效数据表示。先前的研究开发了用于粒子对撞机物理学的OmniLearned基础模型，并表明它可以显著提高对撞机实验的发现潜力。在本文中，我们超越了粒子对撞机物理学，展示了在对撞机数据上训练的基础模型可以帮助改进宇宙学参数的预测，并预测CosmoBench不同数据集中的晕和星系速度。这是第一次有研究表明粒子对撞机物理模型可以跨科学领域泛化。


### 论文摘要

Foundation models build an effective representations of data that can be deployed on diverse downstream tasks. Previous research developed the OmniLearned foundation model for collider physics and showed that it could significantly advance discovery potential across collider experiments. In this paper we go beyond collider physics and show that Foundation Models trained on collider data can help improve the prediction of cosmological parameters and to predict halo and galaxy velocities in different datasets from CosmoBench. This is the first time a collider physics model is shown to generalize across scientific fields.

---

## 82. 论文ID: 2512.24385v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.24385v1.json'

---

## 83. SeedFold: Scaling Biomolecular Structure Prediction

**论文链接:** [http://arxiv.org/abs/2512.24354v1](http://arxiv.org/abs/2512.24354v1)

**作者:** Zhou Yi, Lu Chan, Ma Yiming, Qu Wei, Ye Fei, Zhang Kexin, Wang Lan, Gui Minrui, Gu Quanquan

**发布时间:** 2025-12-30

### GPT解析

### 总结

SeedFold是一个成功的折叠模型，能够有效扩展模型容量，在大多数蛋白质相关任务上优于AlphaFold3。

### 背景

高精度的生物分子结构预测是开发生物分子基础模型的关键组成部分，而构建基础模型的一个最关键方面是确定模型扩展的配方。

### 目的

开发一个能够成功扩展模型容量的折叠模型。

### 方法

1) 确定了一种有效的宽度扩展策略，用于增强Pairformer的表示能力；2) 引入了一种新颖的线性三角注意力机制，降低了计算复杂度，实现了高效扩展；3) 构建了一个大规模的蒸馏数据集，显著扩大了训练集规模。

### 主要发现

在FoldBench上的实验表明，SeedFold在大多数蛋白质相关任务上优于AlphaFold3。

### 结论

SeedFold通过三种创新策略成功实现了模型容量的有效扩展，并在蛋白质结构预测任务中取得了优异性能。

### 翻译

高精度的生物分子结构预测是开发生物分子基础模型的关键组成部分，而构建基础模型的一个最关键方面是确定模型扩展的配方。在这项工作中，我们提出了SeedFold，这是一个成功扩展了模型容量的折叠模型。我们的贡献有三方面：首先，我们确定了一种有效的宽度扩展策略，用于增强Pairformer的表示能力；其次，我们引入了一种新颖的线性三角注意力，降低了计算复杂度，实现了高效扩展；最后，我们构建了一个大规模的蒸馏数据集，显著扩大了训练集。在FoldBench上的实验表明，SeedFold在大多数蛋白质相关任务上优于AlphaFold3。


### 论文摘要

Highly accurate biomolecular structure prediction is a key component of developing biomolecular foundation models, and one of the most critical aspects of building foundation models is identifying the recipes for scaling the model. In this work, we present SeedFold, a folding model that successfully scales up the model capacity. Our contributions are threefold: first, we identify an effective width-scaling strategy for the Pairformer to increase representation capacity; second, we introduce a novel linear triangular attention that reduces computational complexity to enable efficient scaling; finally, we construct a large-scale distillation dataset to substantially enlarge the training set. Experiments on FoldBench show that SeedFold outperforms AlphaFold3 on most protein-related tasks.

---

## 84. Virtual-Eyes: Quantitative Validation of a Lung CT Quality-Control Pipeline for Foundation-Model Cancer Risk Prediction

**论文链接:** [http://arxiv.org/abs/2512.24294v1](http://arxiv.org/abs/2512.24294v1)

**作者:** Md. Enamul Hoq, Linda Larson-Prior, Fred Prior

**发布时间:** 2025-12-30

**备注:** 23 pages, and Under Review-MIDL-2026

### GPT解析

### 总结

本研究开发了Virtual-Eyes，一种针对低剂量CT肺癌筛查的16位CT质量控制管道，并评估了其对通用基础模型和专家模型的差异化影响，发现解剖学定向的质量控制可以改善通用模型性能但可能扰乱专家模型。

### 背景

在低剂量CT肺癌筛查的深度学习管道中，鲁棒的预处理很少被量化，现有方法可能未充分考虑临床需求。

### 目的

开发和验证Virtual-Eyes临床CT质量控制管道，并测量其对通用基础模型与专家模型的差异化影响。

### 方法

开发Virtual-Eyes管道强制执行512x512分辨率，拒绝非诊断系列，使用Hounsfield单位过滤提取肺块；使用765名NLST患者数据，评估RAD-DINO、Merlin、Sybil和ResNet-18模型在原始与预处理输入下的性能。

### 主要发现

Virtual-Eyes显著改善RAD-DINO性能(切片级AUC从0.576到0.610，患者级AUC从0.619到0.735)，校准也提升；而Sybil和ResNet-18性能下降，Merlin仅显示有限改善。

### 结论

解剖学定向的质量控制可以稳定和改善通用基础模型工作流程，但可能扰乱适应原始临床背景的专家模型。

### 翻译

在低剂量CT肺癌筛查的深度学习管道中，鲁棒的预处理很少被量化。我们开发并验证了Virtual-Eyes，这是一个具有临床动机的16位CT质量控制管道，并测量了它对通用基础模型与专家模型的差异化影响。Virtual-Eyes强制执行严格的512x512平面内分辨率，拒绝短或非诊断系列，并使用Hounsfield单位过滤和双侧肺覆盖评分提取连续肺块，同时保留原始16位网格。使用765名NLST患者(182名癌症患者，583名非癌症患者)，我们使用冻结编码器从RAD-DINO和Merlin计算切片级嵌入，并训练无泄漏的患者级MLP头；我们还评估了Sybil和2D ResNet-18基线模型在原始输入与Virtual-Eyes输入下的性能，不重新训练骨干网络。Virtual-Eyes将RAD-DINO的切片级AUC从0.576提高到0.610，患者级AUC从0.646提高到0.683(平均池化)和从0.619提高到0.735(最大池化)，校准也有所改善(Brier分数从0.188降至0.112)。相比之下，Sybil和ResNet-18在Virtual-Eyes下性能下降(Sybil AUC从0.886降至0.837；ResNet-18 AUC从0.571降至0.596)，存在上下文依赖性和捷径学习的证据，而Merlin无论如何预处理都显示出有限的迁移性(AUC从约0.507提高到0.567)。这些结果表明，解剖学定向的质量控制可以稳定和改善通用基础模型的工作流程，但可能会扰乱适应原始临床背景的专家模型。


### 论文摘要

Robust preprocessing is rarely quantified in deep-learning pipelines for low-dose CT (LDCT) lung cancer screening. We develop and validate Virtual-Eyes, a clinically motivated 16-bit CT quality-control pipeline, and measure its differential impact on generalist foundation models versus specialist models. Virtual-Eyes enforces strict 512x512 in-plane resolution, rejects short or non-diagnostic series, and extracts a contiguous lung block using Hounsfield-unit filtering and bilateral lung-coverage scoring while preserving the native 16-bit grid. Using 765 NLST patients (182 cancer, 583 non-cancer), we compute slice-level embeddings from RAD-DINO and Merlin with frozen encoders and train leakage-free patient-level MLP heads; we also evaluate Sybil and a 2D ResNet-18 baseline under Raw versus Virtual-Eyes inputs without backbone retraining. Virtual-Eyes improves RAD-DINO slice-level AUC from 0.576 to 0.610 and patient-level AUC from 0.646 to 0.683 (mean pooling) and from 0.619 to 0.735 (max pooling), with improved calibration (Brier score 0.188 to 0.112). In contrast, Sybil and ResNet-18 degrade under Virtual-Eyes (Sybil AUC 0.886 to 0.837; ResNet-18 AUC 0.571 to 0.596) with evidence of context dependence and shortcut learning, and Merlin shows limited transferability (AUC approximately 0.507 to 0.567) regardless of preprocessing. These results demonstrate that anatomically targeted QC can stabilize and improve generalist foundation-model workflows but may disrupt specialist models adapted to raw clinical context.

---

## 85. 论文ID: 2512.24260v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.24260v1.json'

---

## 86. 论文ID: 2512.24231v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.24231v1.json'

---

## 87. ARM: A Learnable, Plug-and-Play Module for CLIP-based Open-vocabulary Semantic Segmentation

**论文链接:** [http://arxiv.org/abs/2512.24224v1](http://arxiv.org/abs/2512.24224v1)

**作者:** Ziquan Liu, Zhewei Zhu, Xuyang Shi

**发布时间:** 2025-12-30

**备注:** 10 pages, 4 figures

### GPT解析

### 总结

本文提出了一种轻量级的注意力细化模块(ARM)，用于解决开放词汇语义分割(OVSS)中CLIP模型缺乏精确像素级细节的问题。ARM通过学习自适应融合层次化特征，实现了'一次训练，随处使用'的范式，在各种无训练框架中作为即插即用后处理器显著提升性能，同时保持极低的推理开销。

### 背景

开放词汇语义分割(OVSS)受到CLIP模型粗粒度、图像级表示的限制，缺乏精确的像素级细节。现有无训练方法要么依赖昂贵的外部基础模型，要么应用静态的手工启发式方法，导致计算成本高或性能次优。

### 目的

开发一种轻量级、可学习的模块，有效解锁并优化CLIP的内部潜力，解决现有方法在计算效率和性能上的不足。

### 方法

提出注意力细化模块(ARM)，使用语义引导的交叉注意块，利用强大的深度特征(K,V)选择和细化细节丰富的浅层特征(Q)，后接自注意块。ARM在通用数据集上训练一次后，可作为通用即插即用后处理器应用于各种无训练框架。

### 主要发现

大量实验表明，ARM在多个基准测试中一致性地提高了基线性能，且推理开销可忽略不计，为无训练OVSS建立了高效有效的范式。

### 结论

ARM成功解决了CLIP在像素级语义分割中的局限性，通过轻量级设计实现了高性能与高效率的平衡，为开放词汇语义分割提供了新的解决方案。

### 翻译

开放词汇语义分割(OVSS)从根本上受到CLIP粗糙的图像级表示的阻碍，这些表示缺乏精确的像素级细节。现有的无训练方法要么通过导入来自昂贵外部基础模型(如SAM、DINO)的先验知识来解决这个问题，要么对CLIP的内部特征应用静态的、手工制作的启发式方法。这些方法要么计算成本高，要么次优。我们提出了注意力细化模块(ARM)，这是一个轻量级、可学习的模块，能够有效解锁并优化CLIP的内部潜力。与静态融合方法不同，ARM学习自适应地融合层次化特征。它采用语义引导的交叉注意块，利用强大的深度特征(K,V)来选择和细化细节丰富的浅层特征(Q)，然后是自注意块。关键创新在于'一次训练，随处使用'的范式。在通用数据集(如COCO-Stuff)上训练一次后，ARM可以作为各种无训练框架的通用即插即用后处理器。大量实验表明，ARM在多个基准测试中一致性地提高了基线性能，且推理开销可忽略不计，为无训练OVSS建立了高效有效的范式。


### 论文摘要

Open-vocabulary semantic segmentation (OVSS) is fundamentally hampered by the coarse, image-level representations of CLIP, which lack precise pixel-level details. Existing training-free methods attempt to resolve this by either importing priors from costly external foundation models (e.g., SAM, DINO) or by applying static, hand-crafted heuristics to CLIP's internal features. These approaches are either computationally expensive or sub-optimal. We propose the Attention Refinement Module (ARM), a lightweight, learnable module that effectively unlocks and refines CLIP's internal potential. Unlike static-fusion methods, ARM learns to adaptively fuse hierarchical features. It employs a semantically-guided cross-attention block, using robust deep features (K, V) to select and refine detail-rich shallow features (Q), followed by a self-attention block. The key innovation lies in a ``train once, use anywhere" paradigm. Trained once on a general-purpose dataset (e.g., COCO-Stuff), ARM acts as a universal plug-and-play post-processor for diverse training-free frameworks. Extensive experiments show that ARM consistently boosts baseline performance on multiple benchmarks with negligible inference overhead, establishing an efficient and effective paradigm for training-free OVSS.

---

## 88. 论文ID: 2512.24172v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.24172v1.json'

---

## 89. Towards Open-Vocabulary Industrial Defect Understanding with a Large-Scale Multimodal Dataset

**论文链接:** [http://arxiv.org/abs/2512.24160v1](http://arxiv.org/abs/2512.24160v1)

**作者:** TsaiChing Ni, ZhenQi Chen, YuanFu Yang

**发布时间:** 2025-12-30

### GPT解析

### 总结

研究提出了IMDD-1M，首个包含100万对图像-文本对的大规模工业多模态缺陷数据集，并基于此训练了专门针对工业场景的扩散式视觉语言基础模型，该模型通过轻量级微调即可高效适应专业领域，仅需专用专家模型不到5%的数据就能达到 comparable 性能。

### 背景

工业制造领域需要高质量缺陷检测和分类的多模态学习工具，但目前缺乏大规模、多样化的数据集支持。

### 目的

创建并利用IMDD-1M数据集推进制造业和质量检查的多模态学习，并训练一个专门针对工业场景的基础模型。

### 方法

构建IMDD-1M数据集，包含100万对对齐的图像-文本对，涵盖60多种材料类别和400多种缺陷类型，每个都有专家验证的注释和细粒度文本描述。基于此数据集，从头训练一个扩散式视觉语言基础模型，并通过轻量级微调使其适应专业领域。

### 主要发现

该基础模型只需要专用专家模型不到5%的任务特定数据就能达到 comparable 性能，展示了数据高效的领域自适应基础模型在工业检测和生成方面的潜力。

### 结论

IMDD-1M数据集和基于它的基础模型为可扩展、领域自适应和知识驱动的制造智能铺平了道路。

### 翻译

我们提出了IMDD-1M，这是首个大规模工业多模态缺陷数据集，包含100万对对齐的图像-文本对，旨在推进制造业和质量检查的多模态学习。IMDD-1M包含高分辨率真实世界缺陷，涵盖60多种材料类别和400多种缺陷类型，每个都配有专家验证的注释和细粒度文本描述，详细说明缺陷位置、严重程度和上下文属性。该数据集支持广泛的应用，包括分类、分割、检索、字幕生成和生成建模。基于IMDD-1M，我们从零开始训练了一个基于扩散的视觉语言基础模型，专门针对工业场景。该模型作为一个可泛化的基础，可以通过轻量级微调高效适应专业领域。仅需专用专家模型不到5%的任务特定数据，它就能达到 comparable 性能，突显了数据高效的领域自适应基础模型在工业检测和生成方面的潜力，为可扩展、领域自适应和知识驱动的制造智能铺平了道路。


### 论文摘要

We present IMDD-1M, the first large-scale Industrial Multimodal Defect Dataset comprising 1,000,000 aligned image-text pairs, designed to advance multimodal learning for manufacturing and quality inspection. IMDD-1M contains high-resolution real-world defects spanning over 60 material categories and more than 400 defect types, each accompanied by expert-verified annotations and fine-grained textual descriptions detailing defect location, severity, and contextual attributes. This dataset enables a wide spectrum of applications, including classification, segmentation, retrieval, captioning, and generative modeling. Building upon IMDD-1M, we train a diffusion-based vision-language foundation model from scratch, specifically tailored for industrial scenarios. The model serves as a generalizable foundation that can be efficiently adapted to specialized domains through lightweight fine-tuning. With less than 5% of the task-specific data required by dedicated expert models, it achieves comparable performance, highlighting the potential of data-efficient foundation model adaptation for industrial inspection and generation, paving the way for scalable, domain-adaptive, and knowledge-grounded manufacturing intelligence.

---

## 90. Anomaly detection in satellite imagery through temporal inpainting

**论文链接:** [http://arxiv.org/abs/2512.23986v1](http://arxiv.org/abs/2512.23986v1)

**作者:** Bertrand Rouet-Leduc, Claudia Hulbert

**发布时间:** 2025-12-30

### GPT解析

### 总结

该研究展示了如何利用深度学习和卫星时间序列的时间冗余性，通过构建基于SATLAS基础模型的修复模型来检测地表变化，实现了比传统方法更高的敏感性和特异性。

### 背景

从卫星影像检测地表变化对于快速灾害响应和环境监测至关重要，但由于大气噪声、季节性变化和传感器伪影之间的复杂相互作用，这一任务仍然具有挑战性。

### 目的

开发一种能够利用卫星时间序列的时间冗余性，以前所未有的灵敏度检测异常变化的深度学习方法。

### 方法

训练一个基于SATLAS基础模型的修复模型，使用来自不同气候区和土地覆盖类型的全球分布式训练数据，从Sentinel-2时间序列的前期获取数据重建最后一帧。当应用于突然地表变化的区域时，预测与观测之间的差异能揭示传统变化检测方法可能遗漏的异常。

### 主要发现

在2023年土耳其-叙利亚地震序列引发的地震触发地表破裂的验证中，该方法在Tepehan检测到裂隙特征，比时间中位数或Reed-Xiaoli异常检测器具有更高的敏感性和特异性。该方法的检测阈值比基线方法低约三倍。

### 结论

该方法为利用免费可用的多光谱卫星数据进行自动化、全球规模的地表变化监测提供了途径。

### 翻译

从卫星影像检测地表变化对于快速灾害响应和环境监测至关重要，但由于大气噪声、季节性变化和传感器伪影之间的复杂相互作用，这一任务仍然具有挑战性。在这里，我们展示了深度学习如何利用卫星时间序列的时间冗余性，以学习预测在没有变化的情况下地表应该是什么样子的方式，以前所未有的灵敏度检测异常。我们构建了一个基于SATLAS基础模型的修复模型，使用跨越不同气候区和土地覆盖类型的全球分布式训练数据，从前期获取数据重建Sentinel-2时间序列的最后一帧。当应用于突然地表变化的区域时，预测与观测之间的差异揭示了传统变化检测方法可能遗漏的异常。我们在2023年土耳其-叙利亚地震序列引发的地震触发地表破裂上验证了我们的方法，证明在Tepehan检测到裂隙特征比时间中位数或Reed-Xiaoli异常检测器具有更高的敏感性和特异性。我们的方法的检测阈值比基线方法低约三倍，为利用免费可用的多光谱卫星数据进行自动化、全球规模的地表变化监测提供了途径。


### 论文摘要

Detecting surface changes from satellite imagery is critical for rapid disaster response and environmental monitoring, yet remains challenging due to the complex interplay between atmospheric noise, seasonal variations, and sensor artifacts. Here we show that deep learning can leverage the temporal redundancy of satellite time series to detect anomalies at unprecedented sensitivity, by learning to predict what the surface should look like in the absence of change. We train an inpainting model built upon the SATLAS foundation model to reconstruct the last frame of a Sentinel-2 time series from preceding acquisitions, using globally distributed training data spanning diverse climate zones and land cover types. When applied to regions affected by sudden surface changes, the discrepancy between prediction and observation reveals anomalies that traditional change detection methods miss. We validate our approach on earthquake-triggered surface ruptures from the 2023 Turkey-Syria earthquake sequence, demonstrating detection of a rift feature in Tepehan with higher sensitivity and specificity than temporal median or Reed-Xiaoli anomaly detectors. Our method reaches detection thresholds approximately three times lower than baseline approaches, providing a path towards automated, global-scale monitoring of surface changes from freely available multi-spectral satellite data.

---

## 91. Efficient Context Scaling with LongCat ZigZag Attention

**论文链接:** [http://arxiv.org/abs/2512.23966v1](http://arxiv.org/abs/2512.23966v1)

**作者:** Chen Zhang, Yang Bai, Jiahuan Li, Anchun Gui, Keheng Wang, Feifan Liu, Guanyu Wu, Yuwei Jiang, Defei Bu, Li Wei, Haihang Jing, Hongyin Tang, Xin Chen, Xiangzhou Huang, Fengcun Li, Rongxiang Weng, Yulei Qian, Yifan Lu, Yerui Sun, Jingang Wang, Yuchen Xie, Xunliang Cai

**发布时间:** 2025-12-30

**备注:** 10 pages, 3 figures, 3 tables

### GPT解析

### 总结

该研究提出了一种名为LongCat ZigZag Attention (LoZA)的稀疏注意力机制，可将全注意力模型转换为稀疏版本，在长上下文场景中实现显著的速度提升，并开发出能够处理多达100万个token的长上下文基础模型。

### 背景

在长上下文处理场景中，现有全注意力模型存在计算效率低的问题，特别是在预填充密集型和解码密集型任务中。

### 目的

设计一种稀疏注意力方案，以有限计算预算将现有全注意力模型转换为稀疏版本，提高长上下文处理效率，支持长程推理和长视野智能体能力。

### 方法

提出LongCat ZigZag Attention (LoZA)稀疏注意力机制，并在训练中期将其应用于LongCat-Flash模型，开发出LongCat-Flash-Exp长上下文基础模型。

### 主要发现

LoZA在长上下文场景中能够实现显著的速度提升，特别是在预填充密集型（如检索增强生成）和解码密集型（如工具集成推理）情况下；LongCat-Flash-Exp能够快速处理多达100万个token，实现高效的长程推理和长视野智能体能力。

### 结论

LoZA是一种有效的稀疏注意力机制，能够以有限计算预算显著提升长上下文处理效率，为长程推理和长视野智能体应用提供了基础。

### 翻译

我们引入了LongCat ZigZag Attention (LoZA)，这是一种稀疏注意力方案，旨在将任何现有的全注意力模型转换为稀疏版本，且计算预算相当有限。在长上下文场景中，LoZA可以在预填充密集型（如检索增强生成）和解码密集型（如工具集成推理）情况下实现显著的速度提升。具体来说，通过在训练中期将LoZA应用于LongCat-Flash，我们推出了LongCat-Flash-Exp作为长上下文基础模型，能够快速处理多达100万个token，实现高效的长程推理和长视野智能体能力。


### 论文摘要

We introduce LongCat ZigZag Attention (LoZA), which is a sparse attention scheme designed to transform any existing full-attention models into sparse versions with rather limited compute budget. In long-context scenarios, LoZA can achieve significant speed-ups both for prefill-intensive (e.g., retrieval-augmented generation) and decode-intensive (e.g., tool-integrated reasoning) cases. Specifically, by applying LoZA to LongCat-Flash during mid-training, we serve LongCat-Flash-Exp as a long-context foundation model that can swiftly process up to 1 million tokens, enabling efficient long-term reasoning and long-horizon agentic capabilities.

---

## 92. Scaling Remote Sensing Foundation Models: Data Domain Tradeoffs at the Peta-Scale

**论文链接:** [http://arxiv.org/abs/2512.23903v1](http://arxiv.org/abs/2512.23903v1)

**作者:** Charith Wickrema, Eliza Mace, Hunter Brown, Heidys Cabrera, Nick Krall, Matthew O'Neill, Shivangi Sarkar, Lowell Weissman, Eric Hughes, Guido Zarrella

**发布时间:** 2025-12-29

### GPT解析

### 总结

研究人工智能在高分辨率光电数据集上的扩展行为，建立训练大规模基础模型的技术

### 背景

现代多模态机器学习应用依赖非文本模态的稳健领域专用编码器，但在遥感等高价值领域，模型能力、训练计算和数据集规模之间的关系尚不明确

### 目的

建立在高分辨率光电数据集上训练基础模型的实用技术，这些数据集的规模比当前最先进水平高出几个数量级

### 方法

使用超过一千万亿像素的商业卫星光电数据和MITRE联邦AI沙盒，逐步训练更大的视觉变压器主干网络，报告在petascale规模观察到的成功和失败模式

### 主要发现

即使在如此大的规模下，性能仍受限于数据而非模型参数

### 结论

这些见解旨在为数据收集策略、计算预算和优化计划提供信息，以促进未来前沿规模遥感基础模型的发展

### 翻译

我们探索人工智能的扩展行为，以建立在高分辨率光电数据集上训练基础模型的实用技术，这些数据集的规模比当前最先进水平高出几个数量级。现代多模态机器学习应用，如图像标注、搜索和推理的生成式人工智能系统，依赖非文本模态的稳健领域专用编码器。在互联网规模数据丰富的自然图像领域，成熟的扩展定律有助于优化模型能力、训练计算和数据集规模的联合扩展。不幸的是，在遥感等高价值领域中，这些关系不太清楚。使用超过一千万亿像素的商业卫星光电数据和MITRE联邦AI沙盒，我们逐步训练更大的视觉变压器主干网络，报告在petascale规模观察到的成功和失败模式，并分析跨越其他遥感模态领域差距的影响。我们观察到，即使在如此大的规模下，性能仍受限于数据而非模型参数。这些实用见解旨在为数据收集策略、计算预算和优化计划提供信息，以促进未来前沿规模遥感基础模型的发展。


### 论文摘要

We explore the scaling behaviors of artificial intelligence to establish practical techniques for training foundation models on high-resolution electro-optical (EO) datasets that exceed the current state-of-the-art scale by orders of magnitude. Modern multimodal machine learning (ML) applications, such as generative artificial intelligence (GenAI) systems for image captioning, search, and reasoning, depend on robust, domain-specialized encoders for non-text modalities. In natural-image domains where internet-scale data is plentiful, well-established scaling laws help optimize the joint scaling of model capacity, training compute, and dataset size. Unfortunately, these relationships are much less well-understood in high-value domains like remote sensing (RS). Using over a quadrillion pixels of commercial satellite EO data and the MITRE Federal AI Sandbox, we train progressively larger vision transformer (ViT) backbones, report success and failure modes observed at petascale, and analyze implications for bridging domain gaps across additional RS modalities. We observe that even at this scale, performance is consistent with a data limited regime rather than a model parameter-limited one. These practical insights are intended to inform data-collection strategies, compute budgets, and optimization schedules that advance the future development of frontier-scale RS foundation models.

---

## 93. Exploiting the Prior of Generative Time Series Imputation

**论文链接:** [http://arxiv.org/abs/2512.23832v1](http://arxiv.org/abs/2512.23832v1)

**作者:** YuYang Miao, Chang Li, Zehua Chen

**发布时间:** 2025-12-29

### GPT解析

### 总结

本文提出了Bridge-TS模型，通过两种新颖的先验设计（专家先验和组合先验）改进生成式时间序列插值方法，在多个基准数据集上实现了插值精度的突破。

### 背景

时间序列插值在电力、金融和天气建模等领域有广泛应用。先前方法使用扩散概率模型和Schrodinger桥模型等生成模型，但其先验信息对真实目标不够有信息量，导致生成过程负担增加且插值精度有限。

### 目的

提出一种改进的生成式时间序列插值方法，通过改进先验设计来提高插值准确性。

### 方法

Bridge-TS模型构建了数据到数据的生成过程，并采用两种新颖的先验设计：1）专家先验：利用预训练的基于Transformer的模块进行确定性估计作为先验；2）组合先验：结合多个预训练模型的估计结果，实现组合先验到目标的插值过程。

### 主要发现

在ETT、Exchange和Weather等多个基准数据集上的实验表明，Bridge-TS在均方误差和平均绝对误差方面达到了新的插值精度记录。

### 结论

改进先验对于生成式时间序列插值具有优越性，能够显著提高插值准确性。

### 翻译

时间序列插值，即填充时间记录中的缺失值，在电力、金融和天气建模等领域有广泛应用。先前方法引入了生成模型，如扩散概率模型和Schrodinger桥模型，从高斯噪声或线性插值结果条件生成缺失值。然而，由于它们的先验对真实目标不够有信息量，其生成过程不可避免地面临负担增加和插值精度有限的问题。在这项工作中，我们提出了Bridge-TS，为生成式时间序列插值构建了数据到数据的生成过程，并利用两种新颖的设计来设计先验。首先，我们提出专家先验，利用预训练的基于Transformer的模块作为专家，用确定性估计填充缺失值，然后将结果作为真实目标的先验。其次，我们探索组合先验，利用多个预训练模型提供不同的估计结果，然后在数据到数据的生成过程中组合它们，实现组合先验到目标的插值过程。在ETT、Exchange和Weather等几个基准数据集上进行的实验表明，Bridge-TS在均方误差和平均绝对误差方面达到了插值准确性的新记录，证明了改进先验对生成式时间序列插值的优越性。


### 论文摘要

Time series imputation, i.e., filling the missing values of a time recording, finds various applications in electricity, finance, and weather modelling. Previous methods have introduced generative models such as diffusion probabilistic models and Schrodinger bridge models to conditionally generate the missing values from Gaussian noise or directly from linear interpolation results. However, as their prior is not informative to the ground-truth target, their generation process inevitably suffer increased burden and limited imputation accuracy. In this work, we present Bridge-TS, building a data-to-data generation process for generative time series imputation and exploiting the design of prior with two novel designs. Firstly, we propose expert prior, leveraging a pretrained transformer-based module as an expert to fill the missing values with a deterministic estimation, and then taking the results as the prior of ground truth target. Secondly, we explore compositional priors, utilizing several pretrained models to provide different estimation results, and then combining them in the data-to-data generation process to achieve a compositional priors-to-target imputation process. Experiments conducted on several benchmark datasets such as ETT, Exchange, and Weather show that Bridge-TS reaches a new record of imputation accuracy in terms of mean square error and mean absolute error, demonstrating the superiority of improving prior for generative time series imputation.

---

## 94. 论文ID: 2512.23786v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23786v1.json'

---

## 95. 论文ID: 2512.24922v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.24922v1.json'

---

## 96. 论文ID: 2512.24896v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.24896v1.json'

---

