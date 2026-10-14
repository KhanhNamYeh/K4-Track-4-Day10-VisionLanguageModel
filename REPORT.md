Họ tên: Nguyễn Đặng Nam Khánh (MSSV 2A202602741)
`dataset_version`: d10-21e4070301

> Phần cứng: Colab **G4** (NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB), điều khiển bằng `colab` CLI; Python 3.13, torch 2.11, transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2.
> Mọi setup chạy **100 bước tối ưu** (batch 2 × tích luỹ 4 = 8 ảnh/bước, tức 800 ảnh ≈ 0,56 epoch), không phải 300 bước như mặc định của đề. Các siêu tham số khác giữ nguyên `configs/`.
> VRAM là `torch.cuda.max_memory_allocated()` trong lúc train. Thời gian chỉ tính vòng train, không tính nạp model và suy luận.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | – | – | 0,086 | 0,061–0,114 | 0,92 | 0,128 | 0,027 | 0,184 |
| E1 | LoRA LLM | 33.030.144 | 6,44 | 1,78 | 0,242 | 0,210–0,278 | 1,49 | 0,276 | 0,152 | 0,494 |
| E2 | + projector | 142.135.296 | 8,13 | 1,78 | 0,250 | 0,214–0,288 | 1,27 | 0,162 | 0,208 | 0,448 |
| E3 | + LoRA ViT | 148.426.752 | 8,31 | 2,29 | **0,343** | 0,304–0,383 | 1,17 | 0,179 | 0,295 | 0,603 |
| E4 | data_size = 0,1 (cấu hình E3, 142 ảnh) | 148.426.752 | 7,76 | 2,27 | 0,299 | 0,263–0,339 | 1,25 | 0,247 | 0,187 | 0,584 |
| E4 | data_size = 1,0 (= E3) | 148.426.752 | 8,31 | 2,29 | 0,343 | 0,304–0,383 | 1,17 | 0,179 | 0,295 | 0,603 |

`parse_valid_rate`: E0 0,920 · E1 0,974 · E2 0,997 · E3 0,990 · E4(0,1) 1,000. Mean IoU của các cặp khớp: E0 0,598 → E3 0,677.
Loss (trung bình 10 bước cuối): E1 2,05 → 0,64 · E2 2,05 → 0,61 · E3 2,05 → 0,57 · E4(0,1) 2,12 → 0,41. Log từng bước nằm trong `report/runs/*.json`.

### Ảnh minh hoạ (`report/figures/`, box **xanh lá = nhãn đúng**, box **đỏ = dự đoán**)

| Ảnh | Model | TP / FP / FN | Ghi chú |
|---|---|---|---|
| `e3_good_0.jpg` | E3 | 3 / 0 / 0 | Đúng cả 2 pothole và 1 manhole |
| `e3_good_1.jpg` | E3 | 2 / 0 / 0 | 2 manhole: lớp có hình dạng rõ nhất |
| `e3_bad_0.jpg` | E3 | 0 / 2 / 7 | 4 pothole + 3 crack nhỏ, dày đặc: model chỉ trả 2 box và đều lệch |
| `e3_bad_1.jpg` | E3 | 0 / 4 / 5 | Đủ số box nhưng nhầm crack ↔ pothole, IoU < 0,5 |
| `e3_bad_2.jpg` | E3 | 0 / 7 / 1 | Vạch kẻ đường và mép vỉa hè bị gọi là crack; bỏ sót manhole |
| `e3_bad_3.jpg` | E3 | 0 / 4 / 4 | Crack dài bị cắt thành box khác với cách người gán nhãn cắt |
| `e0_good_0.jpg` | E0 | 2 / 0 / 0 | Zero-shot vẫn nhận ra manhole rõ |
| `e0_bad_0.jpg` | E0 | 0 / 3 / 6 | Gọi pothole là crack/manhole, box lớn và lệch |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Tôi đếm tham số bằng `named_parameters()` sau khi nạp model với ViT để ở fp16 (`lab10/sanity.py`). Kết quả:
- Toàn bộ `model.visual` có **415,3M** tham số, gồm 24 khối ViT (≈ 306M), `visual.merger` **27,3M** và `visual.deepstack_merger_list` **81,8M**. Như vậy projector ≈ **109,1M**.
- LLM (36 lớp) cộng embedding ≈ 4,0B, nên tổng ≈ 4,44B. Con số đếm thô ra 2,62B chỉ vì tensor 4-bit của LLM được nén 2 giá trị vào 1 byte (`uint8`).

Phần mỗi setup cập nhật:
- **E1:** chỉ LoRA r=16 trên 252 module của LLM (`q,k,v,o,gate,up,down` × 36 lớp), **33,03M** tham số. ViT và projector đóng băng.
- **E2:** thêm toàn bộ projector (24 tensor), thành **142,14M**, đúng bằng E1 + 109,1M.
- **E3:** thêm LoRA trên 96 module ViT (`attn.qkv, attn.proj, mlp.linear_fc1/2` × 24 khối), thành **148,43M** (+6,29M), tổng cộng 348 module LoRA.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
`parse_valid_rate` = **0,920**: 25/311 output hỏng, và cả 25 đều do **lặp vô hạn** rồi bị cắt ở 256 token. Model không bọc ```json và không thêm chữ thừa, nhờ prompt yêu cầu "ONLY a JSON list". Ví dụ output hỏng:
`[{"label": "crack", "bbox_2d": [559, 370, 600, 400]}, {"label": "crack", "bbox_2d": [570, 400, 600, 430]}, {"label": "crack", "bbox_2d": [570, 430, 600, 460]}, …` (bị cắt giữa chừng).

Hai lỗi phổ biến nhất:
1. **Bịa box crack.** Model trả 213 box crack, nhiều nhất trong 3 lớp, nhưng crack chỉ đạt P = 0,033 và F1 = 0,027. Model gọi vạch kẻ đường, bóng đổ và các ô lát là "crack".
2. **Box sai kích thước hoặc sai vị trí.** Ví dụ `[{"label": "crack", "bbox_2d": [0, 500, 1000, 1000]}]` phủ cả nửa dưới ảnh, trong khi diện tích trung vị của box thật chỉ 0,4–2% khung hình.

Hai lỗi này cho ra precision 0,120 và recall 0,067. Ngược lại, E0 không sai thang: mean IoU của các cặp khớp vẫn đạt 0,598, tức model hiểu thang 0–1000.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Tôi chấm lại chính file `work/e3.jsonl` (model xuất thang 0–1000) với `--coord pixel`. F1 rơi từ **0,343 xuống 0,000**, trong khi `parse_valid_rate` vẫn là 0,990 và FP/ảnh tăng từ 1,17 lên 1,80.

Lý do: khi coi số 0–1000 là pixel, `score.py` nhân x với 1000/640 = 1,56 và y với 1000/360 = 2,78. Mọi box bị kéo giãn và đẩy xuống dưới, nên không cặp nào đạt IoU ≥ 0,5. Kết quả là JSON vẫn hợp lệ nhưng F1 bằng 0, đúng triệu chứng mà GUIDE mô tả.

Qwen3-VL dùng thang 0–1000 vì nó độc lập với độ phân giải: processor resize ảnh động (ở đây ra 220 token), nên toạ độ pixel tuyệt đối sẽ đổi theo kích thước ảnh, còn toạ độ tương đối thì không.

Kiểm chứng phía nhãn: `lab10/sanity.py` đổi 9 box YOLO của một ảnh sang JSON 0–1000 rồi quy ngược về pixel. Sai lệch tối đa là 0,3 px, chỉ do làm tròn.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Mỗi ví dụ dài khoảng 330 token: 220 token `<|image_pad|>`, khoảng 90 token prompt và template, và chỉ 20–150 token JSON. Tôi đặt `labels = -100` cho toàn bộ tiền tố prompt (đã kiểm tra prompt đúng là tiền tố của chuỗi đầy đủ) và cho ký tự `\n` sau `<|im_end|>`. Khi giải mã các token có nhãn, kết quả đúng bằng `[{"label":"crack","bbox_2d":[430,467,583,506]},…]<|im_end|>`.

Nếu tính loss trên cả prompt và ảnh:
- Prompt giống hệt nhau ở mọi ví dụ, nên model sẽ tốn phần lớn gradient để học thuộc nó. Loss giảm nhanh nhưng phần JSON chỉ chiếm một phần nhỏ tín hiệu học.
- Token ảnh là placeholder: nội dung thật đi qua embedding của ViT chứ không phải từ vựng, nên việc dự đoán chúng không có ý nghĩa. Nó chỉ làm loss nhiễu và có thể đẩy model sinh ra `<|image_pad|>`.
- Loss trung bình bị pha loãng. Loss của tôi giảm từ 2,05 xuống 0,57 (E3) chỉ đo phần JSON, nên phản ánh đúng chất lượng grounding.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Trọng số NF4 được lưu dưới dạng `Params4bit` kiểu `uint8` đã nén, kèm `quant_state` (absmax theo khối). Các tensor này không phải số thực, không có `.grad`, nên AdamW không cập nhật trực tiếp được. Muốn sửa chúng chỉ có cách gắn LoRA lên trên.

Tôi gặp đúng lỗi này trong lần chạy đầu. Với transformers 5.19, `llm_int8_skip_modules=["visual","lm_head"]` **không có tác dụng**, vì thư viện khớp bằng `re.match`, tức tính từ đầu tên module, mà tên thật là `model.visual.*`. Kết quả là merger vẫn bị lượng tử hoá:
- Đếm tham số merger ra **13,6M** thay vì 27,3M (bị nén một nửa).
- Phần tăng của E2 chỉ +54,6M thay vì +109M.

Tôi sửa thành `SKIP_MODULES = ["model.visual", "lm_head"]` (`lab10/vlm.py`). Sau đó ViT ở fp16 với 415,3M tham số, và E2 tăng đúng +109,1M.

Projector phải ở float32 vì hai lý do:
1. Với lr 2e-4, mỗi bước cập nhật nhỏ hơn rất nhiều so với độ phân giải của fp16 (≈ 1e-3 tương đối). Phép cộng `w + Δw` trong fp16 bị làm tròn mất, nên tham số gần như không đổi.
2. `GradScaler` từ chối unscale gradient fp16.

Vì vậy tôi đưa tham số projector về `float32` trước khi train. Autocast fp16 vẫn tính forward ở fp16, chỉ có bản trọng số "master" là fp32.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Ảnh trong dataset là frame cắt từ video dashcam. Trong cùng một phiên, hai frame liền kề cách nhau vài giây (ví dụ `vlcsnap-2025-02-18-23h13m55s630` và `…23h13m56s767` đều có trong test), gần như trùng nhau về cảnh và vị trí từng ổ gà.

Nếu chia ngẫu nhiên, frame "anh em" của gần như mọi ảnh test sẽ nằm trong train. Model khi đó chỉ cần nhớ lại box của ảnh gần giống, nên F1 đo được là khả năng ghi nhớ chứ không phải khả năng tổng quát sang đoạn đường mới. F1 sẽ cao giả và không dự báo được hiệu năng thật.

Chia theo phiên thì train có 13 phiên, test có 3 phiên, và không phiên nào nằm ở hai split (đã kiểm tra trong notebook mục 1).

| Split | Ảnh | pothole | crack | manhole |
|---|---|---|---|---|
| train | 1.422 | 1.074 | 1.750 | 691 |
| val | 276 | 66 | 470 | 103 |
| test | 311 | 121 | 299 | 163 |

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **E1 → E2 (mở projector, +109M tham số):** F1 tăng từ 0,242 lên 0,250 (+0,008), hai CI95 gần như trùng hẳn (0,210–0,278 so với 0,214–0,288). VRAM tăng +1,7 GB (6,44 → 8,13), thời gian không đổi (1,78 phút). Lợi ích duy nhất đo được là JSON ổn định hơn (valid 0,974 → 0,997) và FP/ảnh giảm từ 1,49 xuống 1,27. Riêng pothole lại tệ đi (0,276 → 0,162). Kết luận: **không đáng** với 100 bước.
- **E2 → E3 (LoRA ViT, chỉ +6,3M tham số):** F1 tăng từ 0,250 lên **0,343** (+0,093). Cận dưới CI95 của E3 (0,304) cao hơn cận trên của E2 (0,288). Chi phí chỉ +0,18 GB VRAM và +0,5 phút (gradient phải chạy ngược qua 24 khối ViT). Crack tăng từ 0,208 lên 0,295, manhole từ 0,448 lên 0,603.

Kết luận: với bài này, phần đáng tiền nhất là **cho ViT học đặc trưng mới**. Hư hại mặt đường nhỏ và khác xa dữ liệu pretrain, nên chỉ chỉnh projector (vốn chỉ chiếu lại đặc trưng cũ) thì không đủ. E3 là setup đáng đồng tiền nhất: F1 cao nhất mà VRAM và thời gian gần như không đổi.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
CI95 bootstrap theo ảnh rộng khoảng **0,07–0,08 điểm F1**: E1 0,068, E2 0,074, E3 0,079, E4(0,1) 0,076. Hệ quả cho từng phép so sánh:
- **E1 so với E2:** chênh 0,008, nhỏ hơn rất nhiều so với độ rộng CI. Không thể nói projector giúp hay hại.
- **E3 so với E1 và E2:** khoảng cách đủ lớn để hai CI tách nhau, nên E3 tốt hơn là kết luận vững nhất.
- **E4 (0,1 so với 1,0):** chênh 0,044 trong khi hai CI chồng nhau (0,263–0,339 so với 0,304–0,383), nên chưa kết luận được.
- **F1 theo từng lớp còn nhiễu hơn:** test chỉ có 121 pothole, nên pothole dao động 0,16–0,28 giữa các setup mà không theo quy luật nào.

Thêm nữa, test chỉ đến từ **3 phiên quay**. Bootstrap theo ảnh coi các frame là độc lập, trong khi frame cùng phiên tương quan mạnh, nên CI thật còn **rộng hơn** con số báo cáo. Mỗi setup cũng chỉ chạy một seed. Vì vậy tôi chỉ tin các chênh lệch lớn hơn khoảng 0,08 F1.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Ở E3, lớp kém nhất là **pothole (F1 0,179, P 0,194, R 0,165)**, sau đó đến crack (0,295), còn manhole tốt nhất (0,603, P 0,798). Thống kê diện tích box trung vị (% khung hình):

| | train | test |
|---|---|---|
| pothole | 0,38% | 0,42% |
| manhole | 0,39% | 0,67% |
| crack | 1,35% | 1,98% |

- **Pothole:** box rất nhỏ, chỉ khoảng 0,4% ảnh, tức chừng 30×30 px trên ảnh 640×360 và chưa tới 1 patch merge 32×32. Ổ gà xa trông giống vết loang, bóng hoặc crack, nên model nhầm hai chiều crack ↔ pothole (`e3_bad_1.jpg`) và bỏ sót khi có nhiều ổ gà dày đặc (`e3_bad_0.jpg`: 7 FN). Box nhỏ còn khiến IoU ≥ 0,5 khó đạt: lệch 5 px là đủ trượt. Số mẫu không phải nguyên nhân, vì train có 1.074 pothole, nhiều hơn manhole (691).
- **Crack:** box lớn hơn nhưng ranh giới mơ hồ. Một vết nứt dài có thể được người gán nhãn cắt thành 1 hoặc nhiều box, nên dù model khoanh đúng vùng, IoU vẫn thấp (`e3_bad_3.jpg`). Model cũng gọi vạch kẻ đường và mép vỉa hè là crack (`e3_bad_2.jpg`: 7 FP).
- **Manhole:** hình tròn hoặc vuông, viền rõ, ít biến thể. Ngay cả E0 cũng làm đúng (`e0_good_0.jpg`). Lỗi chủ yếu là recall (0,485), do bỏ sót nắp cống xa hoặc bị che (`e3_bad_2.jpg`).

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Tôi đổi đúng **một biến là `data_size`**, trên setup tốt nhất E3 (LoRA LLM + projector + LoRA ViT, r=16, 100 bước, cùng seed): 10% ảnh train (142 ảnh, lấy mẫu ngẫu nhiên seed 42) so với 100% (1.422 ảnh, chính là E3).

| data_size | F1 | CI95 | F1 pothole / crack / manhole |
|---|---|---|---|
| 0,1 | 0,299 | 0,263–0,339 | 0,247 / 0,187 / 0,584 |
| 1,0 | 0,343 | 0,304–0,383 | 0,179 / 0,295 / 0,603 |

100% dữ liệu tốt hơn 0,044 F1, nhưng hai CI95 chồng nhau, nên **chưa đủ bằng chứng** để khẳng định. Kết quả theo lớp còn ngược chiều nhau: crack tốt hơn hẳn khi đủ dữ liệu, còn pothole lại tốt hơn khi chỉ có 10%.

Cũng cần lưu ý rằng với số bước cố định là 100 bước × 8 ảnh:
- Bản 10% thấy mỗi ảnh khoảng **5,6 lần**: loss cuối 0,41, có dấu hiệu bắt đầu thuộc lòng.
- Bản 100% chỉ thấy khoảng **0,56 epoch**: loss cuối 0,57, chưa xem hết dữ liệu.

Như vậy biến "kích thước dữ liệu" thực chất đi kèm số epoch. Điểm có giá trị nhất của thí nghiệm: chỉ 142 ảnh đã đưa F1 từ 0,086 (E0) lên 0,299. Phần lớn lợi ích đến từ việc model học **định dạng và thang toạ độ**, chứ không đến từ lượng dữ liệu. Muốn kết luận chắc hơn cần nhiều seed hoặc test lớn hơn.

## Nhìn lại

- **Lỗi công cụ chấm không bắt được:** `llm_int8_skip_modules=["visual"]` theo GUIDE âm thầm không có tác dụng với transformers 5.19, vì thư viện khớp tên bằng `re.match` tính từ đầu tên module. Train vẫn chạy và F1 vẫn ra số, nhưng projector khi đó là 4-bit và chỉ có 54,6M tham số được "huấn luyện". Tôi chỉ phát hiện nhờ so số tham số với GUIDE (13,6M so với 27,3M). Sau khi bỏ lượng tử hoá ViT, peft lại báo lỗi do `torchao` 0.10 cài sẵn trên Colab, nên tôi phải gỡ gói này.
- **fp16:** ở E3, bước 1 có `grad_norm = NaN`. GradScaler đã bỏ qua bước đó và giảm scale, sau đó loss giảm bình thường. Đây là trường hợp tràn số fp16 đúng như GUIDE cảnh báo, và bị che đi vì không làm loss thành NaN.
- **Nạp lại adapter:** sau khi nạp model gốc, adapter và `projector.pt`, 31/40 output giống hệt lúc suy luận ngay sau train. 9 output khác vì projector được copy về fp16 (lúc train là fp32), nên với greedy decoding một vài token đổi hướng. Nếu nạp lại bằng `PeftModel` mà quên `projector.pt`, kết quả E2/E3 sẽ quay về projector gốc.
- **Điều chưa rõ:** E2 làm pothole tệ đi (0,276 → 0,162) trong khi crack tốt lên. Với 121 pothole và một seed, tôi không phân biệt được đây là nhiễu hay projector đang ưu tiên lớp nhiều mẫu. 100 bước (0,56 epoch) có lẽ là quá ít cho cả ba setup. E5 (GRPO) chưa làm.
