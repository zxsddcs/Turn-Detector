# Turn-Detector

仅通过VAD仅处理音频信息，当对话中出现因思考而产生的停顿情况时，VAD会因检测到停顿而判断句子已结束，但在语义层面上句子尚未完成，由此引入对话轮次检测（turn-detector）。在语义层面上，对话轮次检测能够通过ASR转录后的句子信息，准确判断当前输入是否为成为一个独立的句子。

任务：支持中英文语义级的对话轮次识别，预测next_token为<|im_end|>的概率

模型：经过指令微调后的小参数模型（Qwen2.5-0.5B-Instruct、Qwen3-0.6B）

目标：减少语音对话Pipeline中不准确的VAD打断（如：正在思考下一个词汇时产生的停顿）

模型：
- https://huggingface.co/doodod/Turn-Detector-Qwen3-0.6B
- https://huggingface.co/doodod/Turn-Detector-Qwen2.5-0.5B-Instruct



# 数据集

数据集采用单句文本（Alpaca）、ASR对话文本（Magicdata）等
- Alpaca
- Magicdata
- ShareChatX


## 数据集优化：填充词

调用大模型生成中英文的填充词表
```
en_words = ['uh', 'um', 'ah', 'er', 'hmm', 'mhm', 'uhhuh', 'like', 'you know', 'so', 'actually um', 'basically um', 'actually ah', 'basically ah', 'okay', 'ok']
zh_words = ['嗯', '啊', '哦', '呃', '那个', '这个', '对吧', '是吧', '就是', '然后', '所以', '其实', '反正', '总之', '那个啥', '怎么说呢', '你懂的', '你明白的', '我觉得吧', '我感觉吧', '我认为吧', '我想说', '我想说的是',]
```

## 数据构造示例

```
"埃菲尔铁塔有多高"
"埃菲尔铁塔 嗯... 有多高"
"埃菲尔铁塔 那个 有多高"
```


# 量化

```
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


