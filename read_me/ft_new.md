## Báo cáo tiến độ tinh chỉnh

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
  13/11/2024:
- Vấn đề: Giải quyết vấn đề ngữ nghĩa.

16/11/2024

- What we hav done wrong!
  -> Training on LoRA with not good results in higer rank might be the model you trained have already good enough.
- For the content issues: This problems might be the model not doing well on that information, or the information given to it can be said that a new domains which model haven't seen before... SO it's cost a problems
- Update the training process method notations
