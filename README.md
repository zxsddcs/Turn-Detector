
# Introduction

> Voice Chat Pipeline -> ASR + TurnDetector + VAD + LLM + TTS

In the Voice Chat Pipeline, if we only rely on **VAD (Voice Activity Detection)** to determine whether the user's current turn input has ended, we cannot accurately handle situations where users pause while thinking. When there are pauses during the current turn input that hasn't been completed yet, VAD will detect the pause and prematurely judge that the sentence has ended, but semantically the sentence is not yet complete.

This introduces the **Turn-Detector Model**. The turn detection model is mainly applied in voice + text modal dialogue scenarios. At the semantic level, the turn detection model can analyze the text information transcribed by the ASR model at the semantic level, more accurately determining whether the current user input has ended. The Turn-Detector Model chooses small-parameter (0.5B/0.6B) large models based on Transformer architecture that have undergone instruction fine-tuning, with the main task being to predict the probability of the next_token being `<|im_end|>`.


- **Task**: Semantic-level turn recognition, predicting the probability of next_token being `<|im_end|>`
- **Model**: Small-parameter models after instruction fine-tuning (Qwen2.5-0.5B-Instruct, Qwen3-0.6B)
- **Goal**: Reduce inaccurate VAD interruptions in voice dialogue pipelines (e.g., pauses caused while thinking of the next word)


```
# 1. get the user input
How tall is the Eiffel Tower

# 2. apply_chat_template
<|im_start|>user<|im_sep|>How tall is the Eiffel Tower<|im_end|>

# 3. cut <|im_end|>
<|im_start|>user<|im_sep|>How tall is the Eiffel Tower

# 4. predict next token
```

**HuggingFace Page**
- https://huggingface.co/doodod/Turn-Detector-Qwen3-0.6B
- https://huggingface.co/doodod/Turn-Detector-Qwen2.5-0.5B-Instruct


# Dataset

The turn detection model is mainly applied in Chinese and English voice + text modal dialogue scenarios, with input data types mostly being common text instruction data and colloquial chat dialogue data. Therefore, the dataset uses public datasets such as Alpaca, MagicData (ASR dialogue dataset), ShareChatX, etc.
- Alpaca
- Magicdata
- ShareChatX


Characteristics of ASR transcribed text
- Sometimes sentence endings don't contain punctuation marks
- There may be filler words or ... during the process


Dataset optimization based on ASR transcribed text characteristics
- Sentence filtering: Call large models to analyze current input content, retaining semantically complete and colloquial data from the dataset
- Filler word insertion: Randomly insert 1 filler word in sentences to simulate the actual effect of spoken dialogue

Call large models to generate Chinese and English filler word tables
```python
en_words = ['uh', 'um', 'ah', 'er', 'hmm', ...]
zh_words = ['嗯', '啊', '哦', '呃', '那个', '对吧', ...]
```

Dataset optimization effect example
```json
[
    {
        "instruction": "How tall is the Eiffel Tower",
        "input": "",
        "output": ""
    },
    {
        "instruction": "How tall is the um... Eiffel Tower",
        "input": "",
        "output": ""
    },
    {
        "instruction": "Um how tall is the Eiffel Tower",
        "input": "",
        "output": ""
    },
    ...
]
```


# Quantization

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


# References

## Text
- Think Beyond VAD : End-of-Turn Detection for Voice Bot: A Python Implementation (https://medium.com/@manoranjan.rajguru/end-of-turn-detection-with-transformers-a-python-implementation-23bd74f621f3)
- How to build smarter turn detection for Voice AI (https://blog.speechmatics.com/semantic-turn-detection#wrapping-up-where-were-headed-next)
- LiveKit (https://docs.livekit.io/agents/build/turns/turn-detector/#overview)
- 基于大语言模型实现文本端点检测 (https://blog.csdn.net/qq_33200967/article/details/145210382)


## Audio
- Speculative End-Turn Detector for Efficient Speech Chatbot Assistant (https://arxiv.org/html/2503.23439v1)
- Smart Turn v2 (https://www.daily.co/blog/smart-turn-v2-faster-inference-and-13-new-languages-for-voice-ai/)

