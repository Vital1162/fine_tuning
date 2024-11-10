## Tinh chỉnh mô hình (text compilation)

### Cập nhật:

10/10/2024:

- Lưu ý sau khi tinh chỉnh luôn luôn phải lưu lora adapter
- **Vấn đề**: Chưa giải quyết được vấn đề hiệu suất kém khi sử dụng qlora. Giải pháp đề cử có thể do merged mô hình chưa đúng _Unsloth_ có thể đã merge mô hình Quantization với lora thay vì dequantize trở lại float16/float32 rồi mới merge.
- **Đề cử**:Một vài nguồn tin không được chứng thực cho rằng tăng alpha sẽ tăng hiệu suất mô hình =)). Ví dụ nếu rank = 128 thì alpha nên là 128 (lưu ý có sử dụng rslora).
- **Vấn đề**: Một vài mô hình của unsloth đang rất "?", hãy sử dụng mô hình gốc ví dụ `llama` của `Meta` thay vì của `Unsloth`

- **Phương pháp**:Một vài đề xuất cho rằng thay vì **merge** model thì có thể sử dụng adapter này trên base model hoặc mô hình mà ta tinh chỉnh

11/10/2024:

- **Vấn đề**:1B model instruct tinh chỉnh adapter + base model = Worst model
- **Kết quả**: Model 1B (https://huggingface.co/beyoru/llama3.1_instruct_1B_r256a156ep3_merge_ins) điểm mạnh hạn chế được lỗi sai cấu trúc, điểm yếu MLP ít việc lưu trữ fact hạn chế gây ra việc options đôi khi thiết hợp lý (gọi chung lỗi này là **content issue**).

```
alpha = rank = 256
rslora = True
target_modules = <all_linear>
epochs = 3
learning_rate = 5e-5
merge >> instruct model
```

- **Đề cử**: tăng số lượng tham số cho mô hình dùng mô hình có lượng tham số nhiều hơn giải quyết **content issue**

### Tập dữ liệu sử dụng

MCQ tin học: https://huggingface.co/datasets/beyoru/tong_hop_trac_nghiem?row=0
Tin học MCQ trắc nghiệm

- Contexts: được tạo bằng GPT4o

### Installation

Các cài đặt này có thể thay đổi
Update lại installation: https://github.com/unslothai/unsloth/issues/1144

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

    # added lm_head, embed_tokens with rank 256+ ~ full training
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

Unsloth saving

```
# examples saving as float16
model.push_to_hub_merged("tên_repo", tokenizer, save_method = "merged_16bit", token = "key_hf")
# saving as lora
model.push_to_hub_merged("tên_repo", tokenizer, save_method = "lora", token = "key_hf")
# Don't save as 4 bit
```

Normal saving

```
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token="")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token = '')


!pip install -q peft


from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

adapter_path = ""
config = PeftConfig.from_pretrained(adapter_path)
model_adapter = PeftModel.from_pretrained(model, adapter_path)


merged_model = model_adapter.merge_and_unload()


import gc
import torch
del model_adapter  # Delete the model adapter to free up memory
torch.cuda.empty_cache()  # Clear the CUDA memory cache
gc.collect()  # Collect garbage to further clean up

merged_model.push_to_hub("repo_name", tokenizer, token = "")
```

_Tác động của lượng tử hóa với hiệu suất trong tinh chỉnh qlora_

LoRA trong QLoRA
Trong QLoRA các tham số của LoRA không được lượng tử trực tiếp. Điều này có nghĩa là, LoRA vẫn luôn giữ độ chính xác `fl16/fb16`. Vì vậy các loss trong quá trình tinh chỉnh dựa trên đọ chính xác 16 bit.
Mô hình cơ sở sễ được lượng tử hóa (4/8 bit) cụ thể là lưu trữ dưới dạng NF4/NF8. Nhưng khi tính toán chúng được giải lượng tử hóa lên độ chính xác 16 bits để thực hiện tính toán cũng adapter.

Hợp nhất LoRA với mô hình lượng tử hóa
Sau khi hoàn tất tinh chỉnh và cho tham số của adapter hợp nhất với mô hình cơ sở dạng 4 bits, điều này nghĩa là các tham số của LoRA cũng bị lượng tử hóa, để phù hợp với định dạng ban dữ liệu mô hình cơ sở.
Cấu trúc mô hình sẽ giống hệt so với mô hình chưa tinh chỉnh. Tuy rằng sẽ có độ chính xác cao hơn, nhưng vẫn làm mất đi một phần thông tin do việc chuyển từ 16 bit sang 4 bit.

Suy giảm hiệu suất khi kết hợp
Quá trình lượng tử tham số của LoRA có thể gây ra giảm hiệu suất. Mặc dù các tham số của LoRA được tinh chỉnh ở độ chính xác cao hơn, chúng cũng sẽ không nhận biết được lượng tử hóa. Như đã đề cập ở trên, các tham số của LoRA chỉ thấy các tham số mô hình cơ sở ở định dạng 16-bit chứ không phải định dạng 4-bit.
$W_{LoRA} × W_{base \space 16-bits}$
Và khi kết hợp lại với dạng 4 bits chúng sẽ kết hợp với các tham số mà chúng chưa từng được tinh chỉnh cho, dẫn đến khả năng suy giảm hiệu suất
$W_{merged \space 4-bits} = Quantize(W_{LoRA}) + W_{base \space 4-bits}$

### Trường hợp VRAM không đủ:

- Chọn mô hình khác
- Giảm `per_device_train_batch_size` và `gradient_accumulation_steps`
- Giảm các lớp tinh chỉnh (`lm_head` và `embed_tokens`)
- Giảm rank

### Kiểm tra mô hình

```
# low device
!pip install -q bitsandbytes
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
                            MODEL_NAME
                          )
model = AutoModelForCausalLM.from_pretrained(
                            MODEL_NAME,

                            # load_in_4bit=True # Use this for 8B model above
)

alpaca_prompt = """Sau đây là hướng dẫn mô tả một nhiệm vụ, kết hợp với với hướng dẫn và ngữ cảnh. Hãy viêt một phản hồi là một câu hỏi trắc nghiệm và cung cấp 4 lựa chọn đáp án khác nhau. Hãy chắc chắn rằng mỗi đáp án đều khác biệt, và xác định rõ đáp án đúng.

### Ngữ cảnh
{}

### Phản hổi
{}"""

!pip install -q datasets
## add label and requirements here

contents = ds['context'].to_list()

def generation_df_test(contents: list)
    from tqdm import tqdm
    import pandas as pd
    df = pd.DataFrame(columns=['texts'])
    for content in tqdm(contents)
        prompt = alpaca_prompt.format(content, "")
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=256)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        #answer = answer.replace(prompt, "")
        df.loc[i,'texts'] = answer

    return df

df = generation_df_test(contents)
from google.colab import files
files.download('df.csv')
```

### Cách lựa chọn hệ số phù hợp

Thường là sẽ chẳng có chỉ số nào tuyệt đối cả `rank` và `alpha`.

Theo bài báo `LoRA Learns Less and Forgets Less`. Llama có hiệu suất tốt trong vấn đề toán học. Nhưng không phải tác vụ liên quan đến code, một vài ý kiến chỉ ra rằng bởi khi tinh chỉnh họ đã vô tình bắt mô hình học hỏi kiến thức mới. Điều này sẽ yêu cầu thêm tokens mới vào mô hình.

KHi đưa `rank` của adapter lên quá cao sẽ gây ra _sự sụp đổ gradient_ (`A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA
` có chứng minh về việc này).

Mệnh đề bài báo đưa ra đó là khi tăng kích thước `r`. Khi đó hệ số tỷ lệ của ma trận cấp thấp ($AB, A \in R^{d_1 \times r}, B \in R^{r \times d_2} $)

Khi khởi tạo các phần tử của $B$ thường là 0 và với $A$ sẽ được tạo ngẫu nhiên.

Ở đây hệ số tỷ lệ sẽ là
$\gamma_r$ tỷ lệ nghịch với $r$

Khi tăng $r$ lên thì $\gamma_r$ sẽ giảm đi rất nhiều.

Để tránh điều này khi tinh chỉnh với các $r$ cao. Để đảm bảo sự ổn định khi tăng $r$ thì có nghĩa là ta sẽ làm cho khi tính toán không phụ thuộc vào $r$ nữa =) (Mean và phương sai không phụ thuộc vào rank hay $\Theta_r(1)$)

Lúc này $\nabla_A L$ hay $\nabla_B L$ có độ lớn phục thuộc vào $\frac{\alpha}{\sqrt(r)}$ nhỏ hơn rất nhiều so với $\frac{\alpha}{r}$. Vậy tại sao không phải là $\frac{\alpha}{r^{\frac{1}{4}}}$. Bài báo cũng chứng minh điều này khi xem xét Perplexity.

Về `learning_rate` thì khi điều chỉnh trong $\{5 \times 10^n: -5 < n < -1>\} \cup \{1 \times 10^n: -5 < n <-1\}$ được kiểm tra khi so sánh LoRA với rsLoRA hiệu suất sẽ kém hơn khi so sánh với rsLoRA với default `learning_rate`

Về `alpha` không thấy bất kỳ tài liệu nào đề cập tới. Nhưng ở bài báo `LoRA Learns Less and Forgets Less` họ có sử dụng $\alpha = 32, r= 256$ với tác vụ về lập trình và $\times 2$ với toán học.

_Tóm lại với rank cao thì nên sử dụng rsLoRA hoặc chỉnh alpha thành_ $\alpha = 2*r$

// Notes: CMIIR :v <img src="../img/375d29d5-fbf0-48ae-acfe-19983a14604e.jpeg" alt="Image description" width="10px" height="auto">

## Training on response

Thay vì quan tâm đến loss input thì ta sẽ chỉ quan tâm đến loss của đầu ra.

Ở đây ta sẽ ví dụ với bộ dữ liệu `tong_hop_trac_nghiem`

Các bước Installtion, load model, ... tương tự như trên.

Đối với load dữ liệu thì ta sẽ làm như sau:

```
## tải dữ liệu tong_hop_trac_nghiem
!pip install -q datasets
from datasets import load_dataset

ds = load_dataset("beyoru/tong_hop_trac_nghiem", split='train')
ds.column_names
```

Ta sẽ nhận được kết quả, in ra các cột sau:

```
['question',
 'answer_a',
 'answer_b',
 'answer_c',
 'answer_d',
 'correct_answer',
 'context']
```

### Định dạng lại dữ liệu

Ở đây ta sẽ sử dụng template của llama-3.1/3.2 cũng có thể sử dụng được vì 3.2 chỉ là bản copy của llama 3.1

```
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def format_entry(entry):
    user_content = {'content': entry['context'], 'role': 'user'}

    assistant_content = {
        'content': (
            f"Câu hỏi: {entry['question']}\n"
            f"A. {entry['answer_a']}\n"
            f"B. {entry['answer_b']}\n"
            f"C. {entry['answer_c']}\n"
            f"D. {entry['answer_d']}\n"
            f"Đáp án: {entry['correct_answer']}."
        ),
        'role': 'assistant'
    }

    return {"conversations": [user_content, assistant_content]}

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

formatted_ds = ds.map(format_entry)

from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(formatted_ds)
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

### Training

```
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer), # Thêm dòng này vào
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 8,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

## set up
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

Cuối cùng là train

```
trainer_stats = trainer.train()
```
