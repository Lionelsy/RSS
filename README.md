# 今日论文推荐 - 2025-11-13

共 2 篇论文

---

## 1. Momentor: Advancing Video Large Language Model with Fine-Grained Temporal Reasoning

**论文链接:** [http://arxiv.org/abs/2402.11435v2](http://arxiv.org/abs/2402.11435v2)

**作者:** Long Qian, Juncheng Li, Yu Wu, Yaobo Ye, Hao Fei, Tat-Seng Chua, Yueting Zhuang, Siliang Tang

**发布时间:** 2024-02-18

**备注:** Accepted by ICML 2024

### GPT解析

### 总结

研究提出了一种名为Momentor的Video-LLM，能够完成细粒度的时间理解任务，通过自动数据生成引擎构建了Moment-10M数据集，并在多个任务上证明了其在细粒度时间理解和定位方面的优势。

### 背景

大语言模型在理解和处理基于文本的任务方面表现出色，许多努力正在将这些属性转移到视频模态，称为Video-LLMs。然而，现有的Video-LLMs只能捕获粗粒度语义，无法有效处理与理解或定位特定视频片段相关的任务。

### 目的

解决现有Video-LLMs无法有效处理细粒度时间理解和定位任务的问题，开发一种能够完成细粒度时间理解任务的Video-LLM。

### 方法

提出Momentor，一种能够完成细粒度时间理解任务的Video-LLM；设计一个自动数据生成引擎来构建Moment-10M，这是一个包含片段级指令数据的大规模视频指令数据集；在Moment-10M上训练Momentor，使其能够执行片段级推理和定位。

### 主要发现

Momentor能够在细粒度时间理解和定位任务上表现出色；零样本评估结果表明Momentor在细粒度时间锚定的理解和定位方面具有优势。

### 结论

Momentor代表了一种改进的Video-LLM方法，能够处理细粒度时间理解任务；通过大规模视频指令数据集的训练，模型能够进行片段级推理和定位。

### 翻译

大型语言模型在理解和处理基于文本的任务方面表现出色。许多努力正在将这些属性转移到视频模态，这些被称为Video-LLMs。然而，现有的Video-LLMs只能捕获粗粒度语义，无法有效处理与理解或定位特定视频片段相关的任务。针对这些挑战，我们提出了Momentor，一种能够完成细粒度时间理解任务的Video-LLM。为了支持Momentor的训练，我们设计了一个自动数据生成引擎来构建Moment-10M，这是一个包含片段级指令数据的大规模视频指令数据集。我们在Moment-10M上训练Momentor，使其能够执行片段级推理和定位。在多个任务上的零样本评估表明，Momentor在细粒度时间锚定的理解和定位方面表现出色。


### 论文摘要

Large Language Models (LLMs) demonstrate remarkable proficiency in comprehending and handling text-based tasks. Many efforts are being made to transfer these attributes to video modality, which are termed Video-LLMs. However, existing Video-LLMs can only capture the coarse-grained semantics and are unable to effectively handle tasks related to comprehension or localization of specific video segments. In light of these challenges, we propose Momentor, a Video-LLM capable of accomplishing fine-grained temporal understanding tasks. To support the training of Momentor, we design an automatic data generation engine to construct Moment-10M, a large-scale video instruction dataset with segment-level instruction data. We train Momentor on Moment-10M, enabling it to perform segment-level reasoning and localization. Zero-shot evaluations on several tasks demonstrate that Momentor excels in fine-grained temporally grounded comprehension and localization.

---

## 2. From Everyday to Existential - The ethics of shifting the boundaries of health and data with multimodal digital biomarkers

**论文链接:** [http://arxiv.org/abs/2511.09238v1](http://arxiv.org/abs/2511.09238v1)

**作者:** Joschka Haltaufderheide, Florian Funer, Esther Braun, Hans-Jörg Ehni, Urban Wiesing, Robert Ranisch

**发布时间:** 2025-11-12

**备注:** 11 pages, 2 figures, 1 table

### GPT解析

### 总结

多模态数字生物标志物(MDBs)整合多种生理、行为和环境数据，提供健康的连续表征。MDBs扩展了数字生物标志物的概念维度，包括变异性、复杂性和抽象性，产生了本体论和认识论转变，并对数据驱动的预防医学中的知识、责任和治理产生伦理影响。

### 背景

数字生物标志物在健康监测和预防医学中的应用日益广泛。

### 目的

探讨多模态数字生物标志物如何扩展数字生物标志物的概念，以及这种扩展带来的转变和伦理影响。

### 方法

本文通过理论分析，探讨多模态数字生物标志物的概念扩展及其影响。

### 主要发现

MDBs扩展了数字生物标志物的维度，产生本体论转变(使健康数据化)和认识论转变(重新定义健康相关性)，并带来伦理影响。

### 结论

多模态数字生物标志物的应用代表了健康监测领域的重大转变，需要关注其带来的伦理问题。

### 翻译

多模态数字生物标志物(MDBs)整合多样的生理、行为和环境数据，以提供健康的连续表征。本文认为，MDBs沿着变异性、复杂性和抽象性维度扩展了数字生物标志物的概念，产生了使健康数据化的本体论转变和重新定义健康相关性的认识论转变。这些转变对数据驱动的预防医学中的知识、责任和治理产生了伦理影响。


### 论文摘要

Multimodal digital biomarkers (MDBs) integrate diverse physiological, behavioral, and contextual data to provide continuous representations of health. This paper argues that MDBs expand the concept of digital biomarkers along the dimensions of variability, complexity and abstraction, producing an ontological shift that datafies health and an epistemic shift that redefines health relevance. These transformations entail ethical implications for knowledge, responsibility, and governance in data-driven, preventive medicine.

---

