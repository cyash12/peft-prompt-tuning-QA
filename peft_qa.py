from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, PeftModel, PeftConfig
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset
import sys

def preprocess_function(examples):
    model_name_or_path = "facebook/opt-125m"
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Topic : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == '__main__':
    dataset = load_dataset("yahoo_answers_topics")
    text_column = "best_answer"
    label_column = "topic_text"
    tokenizer_name_or_path = "facebook/opt-125m"
    peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Give the topic: ",
    tokenizer_name_or_path=tokenizer_name_or_path,)
    max_length = 1024
    lr = 3e-2
    num_epochs = 10
    batch_size = 24
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    mode = sys.argv[1]
    if mode == 'train':
        model_name_or_path = "facebook/opt-125m"
        classes = [k.replace("_", " ") for k in dataset["train"].features["topic"].names]
        print(classes)
        dataset = dataset.map(
            lambda x: {"topic_text": [classes[label] for label in x["topic"]]},
            batched=True,
            num_proc=1,
        )
        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",)
        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["train"]
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_dataloader) * num_epochs),
        )
        model = model.to(device)
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            t = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                if t==200:
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                t+=1

            model.eval()
            eval_loss = 0
            eval_preds = []
            t=0
            for step, batch in enumerate(tqdm(eval_dataloader)):
                if t==200:
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
                eval_preds.extend(
                    tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
                )
                t+=1

            eval_epoch_loss = eval_loss / len(eval_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)
            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
        peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
        print(peft_model_id)
        model.save_pretrained(peft_model_id)
        i = 100
        model.eval()
        inputs = tokenizer(f'{text_column} : {dataset["test"][i]["best_answer"]} Topic : ', return_tensors="pt")
        print(dataset["test"][i]["best_answer"])   

        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=20, eos_token_id=3
            )
            print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
    elif mode == 'eval':
        peft_model_id = sys.argv[2]
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, peft_model_id)


        model.to(device)
        model.eval()
        #i = 200
        prompt = sys.argv[3]
        #inputs = tokenizer(f'{text_column} : {dataset["test"][i]["best_answer"]} Topic : ', return_tensors="pt")
        inputs = tokenizer('best_answer: ' + prompt + ' Topic: ', return_tensors="pt")

        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
            )
            print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

    
    
