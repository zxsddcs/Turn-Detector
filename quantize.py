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