# peft-prompt-tuning-QA
Efficient fine-tuning of a large language model

HuggingFace PEFT library provides methods to efficiently fine-tune large language models to perform specific tasks using a variety of methods. In this code, prompt tuning is utilized. The dataset used is the wiki_qa dataset from HuggingFace. It trains only a small percentage of parameters (~ 0.005% for facebook/opt-125m).<br>
To train the model, execute the peft_topics.py file with 'train' as a command line argument.<br>
To evaluate the model, execute the peft_topics.py file with 'eval', saved model name, and a prompt as command line arguments
