import json
import time
import numpy as np
import torch
from loguru import logger
from optimum.onnxruntime import ORTModelForCausalLM
from sklearn.utils.extmath import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM


MAX_HISTORY_TOKENS = 1024
MAX_HISTORY_TURNS = 10


class TurnDetector:
    """
    TurnDetector is a model that can detect the turn of a conversation.
    """

    def __init__(self, model_path):
        self.model = ORTModelForCausalLM.from_pretrained(model_path, use_cache=False)
        # self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.eou_index = self.tokenizer.encode("<|im_end|>")[-1]
        logger.info(f"End of utterance token: {self.eou_index}")
        logger.info(f"End of utterance token: {self.tokenizer.decode([self.eou_index])}")

    def _format_chat_ctx(self, chat_ctx: dict):
        new_chat_ctx = []
        for msg in chat_ctx:
            content = msg.get("content", "").strip()
            if not content:
                continue
            new_chat_ctx.append({"role": msg["role"], "content": content})

        convo_text = self.tokenizer.apply_chat_template(
            new_chat_ctx,
            add_generation_prompt=False,
            add_special_tokens=False,
            tokenize=False,
            enable_thinking=False,
        )
        # Remove the last <|im_end|>
        ix = convo_text.rfind("<|im_end|>")
        text = convo_text[:ix]
        return text
        

    def run(self, data: bytes) -> dict:
        data_json = json.loads(data)
        chat_ctx = data_json.get("chat_ctx", None)
        if not chat_ctx:
            raise ValueError("chat_ctx is required on the inference input data")

        start_time = time.perf_counter()
        text = self._format_chat_ctx(chat_ctx)
        inputs = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=MAX_HISTORY_TOKENS,
            truncation=True,
        ).to(self.model.device)
        # EOU probability
        with torch.no_grad():
            outputs = self.model(**inputs).logits.detach().cpu().numpy()
        logits = outputs[0, -1, :]
        probs = softmax(logits[np.newaxis, :])[0]
        eou_probability = float(probs[self.eou_index])
        end_time = time.perf_counter()

        result = {
            "eou_probability": eou_probability,
            "input": text,
            "duration": round(end_time - start_time, 3),
        }
        return result


if __name__ == '__main__':
    turn_detector = TurnDetector("doodod/Turn-Detector-Qwen3-0.6B")
    threshold = 0.1
    testset = [
        {"role": "user", "content": "你叫什么"},
        {"role": "user", "content": "你叫什么名字"},
        {"role": "user", "content": "找出两个"},
        {"role": "user", "content": "找出两个可以找到化石的地方"},
        {"role": "user", "content": "What is the capital"},
        {"role": "user", "content": "What is the capital of China"},
    ]
    # Single-turn conversation test
    for item in testset:
        data = json.dumps({"chat_ctx": [item]}).encode()
        result = turn_detector.run(data)
        print(result)
