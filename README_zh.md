
# 简介

> Voice Chat Pipeline：ASR + TurnDetector + VAD + LLM + TTS

在Voice Chat Pipeline中，若仅通过**VAD（Voice Activity Detection）**来判断用户当前轮次的输入是否结束，则无法准确处理用户因思考而产生停顿的情况。当对话中出现当前轮次输入尚未完成，但表述过程中存在停顿的情况时，VAD会因检测到停顿而提前判断句子已结束，但在语义层面上句子尚未完成。

由此引入**对话轮次检测模型（Turn-Detector）**。对话轮次检测模型主要应用于语音+文本模态的对话场景。在语义层面上，对话轮次检测模型能够在语义层面上分析ASR模型转录后的文本信息，更加准确地判断当前用户输入是否结束。Turn-Detector Model 选择基于 Transformer 架构且经过指令微调的小参数（0.5B/0.6B）大模型进行训练，主要任务为预测next_token为`<|im_end|>`的概率。

- **任务**：语义级的对话轮次识别，预测next_token为`<|im_end|>`的概率
- **模型**：经过指令微调后的小参数模型（Qwen2.5-0.5B-Instruct、Qwen3-0.6B）
- **目标**：减少语音对话Pipeline中不准确的VAD打断（如：正在思考下一个词汇时产生的停顿）

```
# 1. get the user input
埃菲尔铁塔有多高

# 2. apply_chat_template
<|im_start|>user<|im_sep|>埃菲尔铁塔有多高<|im_end|>

# 3. cut <|im_end|>
<|im_start|>user<|im_sep|>埃菲尔铁塔有多高

# 4. predict next token
```

**HuggingFace Page**
- https://huggingface.co/doodod/Turn-Detector-Qwen3-0.6B
- https://huggingface.co/doodod/Turn-Detector-Qwen2.5-0.5B-Instruct


# 数据集

对话轮次检测模型主要应用于中英文语境下的语音+文本模态的对话场景，输入数据类型多为常用文本指令数据和口语化聊天对话数据。因此，数据集采用了Alpaca、MagicData（ASR对话数据集）、ShareChatX等公开数据集。
- Alpaca
- Magicdata
- ShareChatX


ASR识别后的文本特点
- 有时句尾不包含标点符号
- 过程中会有语气词或...


基于ASR识别后的文本特点进行数据集优化
- 句子过滤：调用大模型分析当前输入内容，保留数据集中语义完整、口语化的数据
- 填充语气词：在句子中随机插入1个语气词，模拟口语对话的实际效果

调用大模型生成中英文的填充词表
```python
en_words = ['uh', 'um', 'ah', 'er', 'hmm', ...]
zh_words = ['嗯', '啊', '哦', '呃', '那个', '对吧', ...]
```

数据集优化效果示例
```json
[
    {
        "instruction": "埃菲尔铁塔有多高",
        "input": "",
        "output": ""
    },
    {
        "instruction": "埃菲尔铁塔 嗯... 有多高",
        "input": "",
        "output": ""
    },
    {
        "instruction": "那个 埃菲尔铁塔 有多高",
        "input": "",
        "output": ""
    },
    ...
]
```


# 量化

```python
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
from transformers import AutoTokenizer

model_checkpoint = ""
save_directory = ""

ort_model = ORTModelForCausalLM.from_pretrained(
    model_checkpoint, 
    export=True,
    use_cache=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
ort_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
quantizer = ORTQuantizer.from_pretrained(ort_model)
quantizer.quantize(save_dir=save_directory, quantization_config=qconfig)
```


# 参考

## Text
- Think Beyond VAD : End-of-Turn Detection for Voice Bot: A Python Implementation (https://medium.com/@manoranjan.rajguru/end-of-turn-detection-with-transformers-a-python-implementation-23bd74f621f3)
- How to build smarter turn detection for Voice AI (https://blog.speechmatics.com/semantic-turn-detection#wrapping-up-where-were-headed-next)
- LiveKit (https://docs.livekit.io/agents/build/turns/turn-detector/#overview)
- 基于大语言模型实现文本端点检测 (https://blog.csdn.net/qq_33200967/article/details/145210382)


## Audio
- Speculative End-Turn Detector for Efficient Speech Chatbot Assistant (https://arxiv.org/html/2503.23439v1)
- Smart Turn v2 (https://www.daily.co/blog/smart-turn-v2-faster-inference-and-13-new-languages-for-voice-ai/)

