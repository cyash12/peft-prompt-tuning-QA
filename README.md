# peft-prompt-tuning-QA
Efficient fine-tuning of a large language model

HuggingFace PEFT library provides methods to efficiently fine-tune large language models to perform specific tasks using a variety of methods. In this code, prompt tuning is utilized. The dataset used is the wiki+qa dataset from HuggingFace. It trains only a small percentage of parameters (~ 0.005% for facebook/opt-125m).
