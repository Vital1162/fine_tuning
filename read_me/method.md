### Methods

#### Vấn đề khi tinh chỉnh:

- Yêu cầu câu hỏi trắc nghiệm chất lượng cao.
- Yêu cầu phần cứng lớn (rất hạn hẹp chủ yếu là tinh chỉnh trên T4) khi tinh chỉnh (FFT).
- Dữ liệu ít và mới là một vấn đề.
- Chất lượng dữ liệu cũng có thể là một vấn đề gặp phải khi tinh chỉnh đặc biệt là khi lượng mẫu ít thì chất lượng là điều vô cùng quan trọng.

#### Sơ bộ

Transformers đã trở thành một kiến trúc nền tảng cho nhiều LLMs. Sử dụng kiến trúc. Sử dụng kiến trúc mã hóa và giải mã (Encoder-Decoder). Bên trong mỗi bộ giải mã hay mã hóa đều tồn tại một lớp multi-head attention (1). Trong đó mỗi đầu đều chứa một số $h$ các hàm self-attention (2). Các vector $Q, K, V \in R^{n \times d}$ là sự biến đổi của chuỗi đầu vào. Về vai trò có thể nói $Attn_{Q, K,V}$ cho phép mô hình tập trung vào dữ liệu đồng thời và nắm bắt được mối quan hệ phức tạp của từng từ trong chuỗi.
**<p style="text-align:center;">$MHA(Q, K, V) = Concat(head_{1}, head\_{2},...,{head_h})W^{O}$ (1)</p>**

**<p style="text-align:center;">$Attn(Q, K, V) = Softmax(\frac{QK^{T}}{\sqrt{d}})$ (2)</p>**

Nên nhớ rằng LLMs khi dự đoán tokens tiếp theo trong một câu $S = \{t_1,t_2, ... t_m\}$ và mô hình dự đoán chính xác từ tiếp theo $t_{m+1}$. Điều đó ngụ ý rằng $t_{m+1}$ đã được tạo ra ở đâu đó trong hàng trăm hoặc tỷ tham số. Tức mô hình đó đã được huấn luyện để có thể ghi nhớ rất nhiều thông tin khác nhau. Và tất cả các thông tin này sẽ được lưu trữ ở MLPs sau quá trình huấn luyện. Nhớ rằng $Attn$ sẽ là lớp mà thông tin được trao đổi giữa các embedding token vector giao tiếp với nhau. Và chúng cũng không đơn thuần đại diện cho một token. Và lớp MLPs cũng chiếm phần lớn số lương tham số trong một LLMs.

Trong quá trình đào tạo trước LLMs sẽ được huấn luyện bằng cách cố gắng hoàn thành một đoạn văn ngẫu nhiên được góp nhặt ở trên mạng. Lúc này mô hình chưa thực sự tốt nếu như mục đích là tạo ra được một AI tốt. Để có thể tiếp tục cải thiện LLMs sẽ phải trải qua một quá trình gọi là tinh chỉnh, các kỹ thuật tinh chỉnh hiện nay rất phổ biến và nhiều nhưng chúng đều có đặc điểm chung đều là SFT. Và cũng có kỹ thuật sau SFT như RLHF, khi mà con người sẽ đánh giá những phản hồi và sửa đổi, sau đó tinh chỉnh để thay đổi tham số giúp cải thiện chất lượng phản hồi.

### Tinh chỉnh

Hiện nay có rất nhiều các mô hình mã nguồn mở được đào tạo trước giúp ta không cần phải tạo ra một mô hình mới từ đầu, thứ sẽ tốn nhiều tài nguyên tính toán cũng như là chi phí cho thiết bị và nhân công.

Tinh chỉnh cũng sẽ vô cũng tốn kém khi số lượng tham số của các mô hình càng ngày nhân rộng để cải thiện hiệu suất. Các mô hình nhỏ (below 8B) sẽ là lựa chọn có thể nói là tối ưu nhất với phần cứng yếu. Nhưng vẫn rất lớn với khi chỉ sử hữu một GPUs trong quá trình huấn luyện. Giả sử với `Llama 3.1 8B` để FFT mô hình, ta sẽ cần tới ít nhất là 40 GB VRAM (SGD) để tinh chỉnh hoàn toàn. Một lý do khác để không sử dụng FFT đó chính là lương mẫu dữ liệu quá ít có thể dẫn tới tinh chỉnh không hiệu quả và khiến giảm hiệu suất mô hình.

### PEFT (LoRA/QLoRA/DoRA)

LoRA được cho là một giải pháp thay thế tốt cho FFT, khi thay vì tinh chỉnh toàn bộ tham số thì ta sẽ chỉ tinh chỉnh một phần nhỏ tham số của mô hình.
**<p style="text-align:center;">$W_{lora} = W*{pretrained} + \Delta W (AB, A \in R^{m \times r}, B \in R^{r \times n} )$</p>**

<update...>
/QLoRA

/DoRA

### Lựa chọn tham số r và $\alpha$

=> Tại sao quan trọng
update...

Măc dù dữ liệu sẽ ảnh hướng lớn tới hiệu suất nhưng các tham số này cũng quan trọng khi mà lựa chọn tham số sai lệch sẽ dẫn tới kết quả không mong muốn.
Vậy tại sao $\alpha$ lại ảnh hưởng tới hiệu suất khi tinh chỉnh. <ngắn gọn> Bởi vì $\alpha$ sẽ được sử dụng là _tốc độ học của LoRA_</ngắn gọn>
**<p style="text-align:center;">$LR_{LoRA} = \frac{\alpha}{{r}} \times LR$</p>**

Trong tinh chỉnh ta sẽ muốn cập nhật tham số trong LoRA một cách nhanh chóng hơn nguyên nhân là bởi dữ liệu tinh chỉnh thường ít hơn rất nhiều với dữ liệu huấn luyện trước. Một lựa chọn trực giác ở đây là sẽ luôn giữ cho $\frac{\alpha}{r} \geq 1 $. Sự ảnh hưởng của rank thì thường rõ rệt hơn so với $\alpha$ khi mà rank sẽ tỷ lệ với số lượng tham số đào tạo trong LoRA.

=> Cái bài báo liên quan đến lựa chọn các tham số
_LoRA vs full finetuning: An illusion of equivalence_ (1)
_A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA_ (2)
_LoRA learn less forgets less_ (3)

**Các kết luận**:

- $\alpha = 2r$ (1)
- Không tinh chỉnh với rank thấp ($\leq 8$) (1)
- Tinh chỉnh không đúng cách sẽ dẫn tới kết quả sai lệch (thiếu lớp trong gate_prj) (2)
- Cẩn thận khi lựa chọn rank ảnh hưởng tới hiệu suất mô hình (2)

Nhận xét: **_Tinh chỉnh thì nên đọc_**
Thực sự không có nguyên tắc nào kể cả khi họ chỉ chứng minh điều đó trên các tập dữ liệu. Khi đặt $\alpha = 2$ tức là họ đang tăng tốc độ học cơ bản lên 2 lần và thực sự thiếu căn cứ và không phù hợp với mô hình có kích thước khác nhau. Đề xuất ở đây đó chính là cho $\alpha$ là một hằng số có thể bằng $\sqrt{dim_{hidden \space layer}}$. Vì khi rank tiệm cận với số chiều của mô hình (với 8B là 4096)

Chứng minh:
$\alpha = \sqrt{dim}$

rsLoRA => lr_lora = $\frac{\sqrt{dim}}{\sqrt{r}} \times lr$

Notes:
$dim >> r$
Điều này giữ tốc độ học đủ lớn, phù hợp với các không gian thấp rank vốn ít thông tin.

rank tiến về dim thì hệ số trên sẽ tiến về 1 => làm tốc độ học đồng bộ với tốc độ học ban đầu của lr

So sánh:
| Kích thước | alpha = 2\*rank | alpha = \sqrt{dim} |
|-------------|----------------------------------|------------------------------|
| Rank nhỏ | Có thể làm tốc độ học quá thấp | Tốc độ học được giữ đủ lớn nhờ|
| Rank cao | Không kiểm soát tốt tốc độ học | Tốc độ học giảm dần, ổn định |
