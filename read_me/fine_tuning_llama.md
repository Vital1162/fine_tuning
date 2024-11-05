## Tinh chỉnh mô hình

### Installation

Các cài đặt này có thể thay đổi

```
%%capture
!pip install unsloth
# Also get the latest nightly Unsloth!
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip uninstall transformers -y && pip install --upgrade --no-cache-dir "git+https://github.com/huggingface/transformers.git"
```

### Model

Tạm thời cứ huấn luyện với 1B

```
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None
load_in_4bit = True
"""
if using transformers:
    the same as using qlora
"""

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit", # Model here
    max_seq_length = max_seq_length,
    dtype = dtype, # for T4 is default to float16
    load_in_4bit = load_in_4bit,
)
```

### PEFT

Điều chỉnh rank hay alpha tùy ý

```
"""
if rslora set to True
    make sure alpha/sqrt(r) >=1

'lm_head', 'embed_tokens' not an good optional
requires more than just T4
"""

model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # 256, 512,...
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0.0, # Don't add dropout to lora
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    use_rslora = True,
    loftq_config = None,
)
```

### Prepare datasets

```
!pip install -q datasets
from datasets import load_dataset

ds = load_dataset("beyoru/tin_hoc", split='train') # tin hoc
#ds = load_dataset("beyoru/tong_hop_trac_nghiem", split='train') # trac nghiem tong hop
ds = ds.to_pandas()
ds
```

### Format dataset

Lưu ý 2 bộ dữ liệu có tên trường khác nhau nên nếu train bộ nào thì sử dụng sửa lại tên cột.

Có thể sửa lại `prompt` nếu muốn

```
prompt = """Sau đây là hướng dẫn mô tả một nhiệm vụ, kết hợp với với hướng dẫn và ngữ cảnh. Hãy viết một phản hồi là một câu hỏi trắc nghiệm và cung cấp 4 lựa chọn đáp án khác nhau. Hãy chắc chắn rằng mỗi đáp án đều khác biệt, và xác định rõ đáp án đúng.

### Ngữ cảnh
{}

### Phản hổi
{}"""
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts }


inputs = ds['para'].tolist()
outputs = ds.apply(lambda row: f"Câu hỏi: {row['question']}\nA.{row['answer_a']}\nB.{row['answer_b']}\nC.{row['answer_c']}\nD.{row['answer_d']}\nĐáp án: {row['answer_key']}", axis=1)

examples = {
    "input": inputs,
    "output": outputs
}

import pandas as pd

formatted_data = formatting_prompts_func(examples)
formatted_df = pd.DataFrame(formatted_data)

from datasets import Dataset

dataset = Dataset.from_pandas(formatted_df)
print(dataset['text'][4])
```

### Huấn luyện

```
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

## For SFT use this (no lm_head, emb_tokens)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(


        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        num_train_epochs = 3, # set this to what you wants
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)


## For CPT use this ()
# trainer = UnslothTrainer(
#     model = model,
#     tokenizer = tokenizer,
#     train_dataset = dataset,
#     dataset_text_field = "text",
#     max_seq_length = max_seq_length,
#     dataset_num_proc = 8,

#     args = UnslothTrainingArguments(
#         per_device_train_batch_size = 4,
#         gradient_accumulation_steps = 8, #4

#         warmup_ratio = 0.1,
#         num_train_epochs = 3, #set what you want

#         # Notes embedding_learning_rate = learning_rate/10 or learning_rate/2
#         learning_rate = 2e-5,
#         embedding_learning_rate = 2e-6,

#         fp16 = not is_bfloat16_supported(),
#         bf16 = is_bfloat16_supported(),
#         logging_steps = 10,
#         optim = "adamw_8bit",
#         weight_decay = 0.01,
#         lr_scheduler_type = "linear",
#         seed = 3407,
#         output_dir = "outputs",
#         report_to = "none",
#     ),
# )


trainer_stats = trainer.train()

## if the above train() not working try this, if not working...
# from unsloth import unsloth_train
# trainer_stats = unsloth_train(trainer)
```

### Saving_model and push to hf

```
# examples saving as float16
model.push_to_hub_merged("pnpm12/mcq_khtn", tokenizer, save_method = "merged_16bit", token = "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")
# saving as lora
model.push_to_hub_merged("pnpm12/mcq_khtn", tokenizer, save_method = "lora", token = "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")
# Don't save as 4 bit
```

Trường hợp VRAM không đủ:

- Chọn mô hình khác
- Giảm `per_device_train_batch_size` và `gradient_accumulation_steps`
- Giảm các lớp tinh chỉnh (`lm_head` và `embed_tokens`)
- Giảm rank