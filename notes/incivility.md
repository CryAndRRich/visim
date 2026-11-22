# Thiếu văn minh (Tấn)

## 1. Dữ liệu

- `dev.csv`: 1.100 bình luận VnExpress. Trước khi rà soát có 930 nhãn `0` và 170 nhãn `1`. Sau lượt 5 có 915 nhãn `0` và 185 nhãn `1`. Phân theo `parent_text`, có 407 bình luận gốc với 33 nhãn `1` và 693 bình luận trả lời với 152 nhãn `1`.
- `train.csv`: 10.000 bình luận UIT-ViCTSD với nhãn **toxic** làm nhãn thay thế cho thiếu văn minh. Loại hai bình luận trùng văn bản với mục cần chấm trong dev sau khi chuẩn hóa Unicode, hoa thường và khoảng trắng. Còn 9.998 mẫu, trong đó 1.100 mẫu dương. Hai bộ dữ liệu dùng định nghĩa khác nhau, nhất là với các câu thể hiện thái độ coi thường mà không dùng từ tục.
- `corpus.csv`: 20.468 bình luận, gồm 12.977 bình luận gốc và 7.491 câu trả lời. ID của dev dùng bí danh khác với ID của corpus. Ghép theo cặp nguyên văn `(text, parent_text)` được 1.098 trường hợp duy nhất và hai nhóm có nội dung giống hệt nhau. Khi xuất nhãn, hai nhóm dùng cùng nhãn trên mọi bản giống hệt. Tổng cộng 1.103 dòng corpus nhận nhãn từ dev. Cột `score` vẫn lấy từ model.
- Dev được lấy mẫu tăng tỷ lệ dương. Tỷ lệ dương của dev dùng để mô tả mẫu đánh giá.

## 2. Các lượt rà soát nhãn

Mục 4 của `TASK.md` quy định người rà soát nhãn. Theo yêu cầu của Tấn, trợ lý AI thực hiện các lượt trong bảng bằng cách đọc từng bình luận và ngữ cảnh, quyết định nhãn theo mục 6 rồi mới đối chiếu nhãn Claude ban đầu. Mỗi lượt dùng một phiếu ẩn nhãn và điểm cũ gồm 10 điểm cao, 10 điểm thấp, 10 điểm gần ngưỡng lấy mẫu. Các lượt loại toàn bộ ID đã xem trước đó.

| Lượt | Cao / thấp / phân vân | 0 → 1 | 1 → 0 | Commit |
|---|---:|---:|---:|---|
| 1 | 10 / 10 / 10 | 6 | 2 | `48b0fdf` |
| 2 | 10 / 10 / 10 | 4 | 0 | `44bc371` |
| 3 | 10 / 10 / 10 | 2 | 1 | `1d4a091` |
| 4 | 10 / 10 / 10 | 4 | 0 | `2932250` |
| 5 | 10 / 10 / 10 | 2 | 0 | `3db6306` |

- Lượt 1: `R222` và `E205` đổi thành `1` vì có từ hạ thấp người phạm tội. `R140` đổi thành `0` vì đưa tình tiết để phản biện. `391` đổi thành `0` vì lời lên án hướng vào hành vi giết trẻ em. `060` kết hợp lý lẽ với câu hỏi tu từ coi thường người đưa ý kiến nên gán `1`.
- Lượt 2: `R010` và `R273` đổi thành `1` vì có câu giễu cợt hướng vào một người hoặc một nhóm. `431` gán `1` vì mỉa mai nhóm tài xế. Trường hợp `023` gán `0` vì “cái này” không đủ để xác định người bị nhắm tới.
- Lượt 3: `153` và `R099` đổi thành `1` vì người viết dùng giọng coi thường ý kiến hoặc năng lực của người phê bình. `R263` đổi thành `0` vì trọng tâm là cảnh báo nguy cơ bạo lực trong một số mối quan hệ.
- Lượt 4: `E131` chế giễu nhận định của người đối thoại. `E184` quy chụp cả một nhóm người dựa trên tình trạng sức khỏe. Cả hai đổi từ `0` sang `1`. Cụm “quá độc ác” ở `190` đứng riêng nên chưa xác định được người bị công kích và giữ nhãn `0`.
- Lượt 5: `E055` quy kết người gây án vốn có bản chất bạo lực. `R230` gán cho nhóm người vi phạm đặc điểm thiếu hiểu biết. Cả hai đổi từ `0` sang `1`.

Dừng sau lượt 5 theo tiêu chí tối đa khoảng hai nhãn sửa trong một lượt 30 mẫu. Tổng cộng đã xem 150 ID khác nhau và sửa 21 nhãn (18 từ `0` sang `1`, ba theo chiều ngược lại). 950 mẫu ngoài các lượt này giữ nhãn Claude ban đầu. Git diff cho thấy nội dung, thứ tự và định dạng CSV được giữ nguyên qua năm lượt.

## 3. Model và thiết kế so sánh

| Cấu hình | Dữ liệu và đầu vào | Thiết lập |
|---|---|---|
| Đa số | Tỷ lệ từ UIT-ViCTSD | Dự đoán lớp `0`, điểm cố định bằng tỷ lệ dương trong train |
| Từ khóa | Bình luận | Danh sách từ miệt thị, từ tục và cụm coi thường. Điểm quy tắc `1 − 2^(-số lần khớp)` |
| TF-IDF + Logistic Regression | UIT-ViCTSD, dev, hoặc cả hai. Thêm hoặc bỏ bình luận cha | N-gram từ 1–2 và ký tự 2–5, `min_df=2`, `sublinear_tf=True`. `class_weight=balanced`, `liblinear`, `C` trong 0,25 / 1 / 4 |
| PhoBERT | UIT-ViCTSD, văn bản bình luận tách từ bằng PyVi | Base v2. AdamW, LR 2e-5, batch 32, 3 epoch, tối đa 192 token, trọng số lớp nghịch tần suất. Lấy trung bình xác suất từ ba lần chạy với seed 42, 43, 44. [Phiên bản PhoBERT](https://huggingface.co/vinai/phobert-base-v2/commit/86cd7fd4c148980922ac11a2cf5e257f2ba639e1) |
| Qwen2.5 7B Instruct | Bình luận và bình luận cha; định nghĩa ở mục 6 | NF4 với tính toán FP16. Zero-shot, 8 ví dụ từ UIT-ViCTSD (4 mỗi lớp), và cùng 8 ví dụ với hướng dẫn suy luận. Seed 42, giải mã tham lam, tối đa 64 token suy luận. Lấy xác suất token `1` chuẩn hóa trên hai token nhãn. [Phiên bản Qwen](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/commit/a09a35458c702b33eeacc393d103063234e8bc28) |

TF-IDF dùng toàn bộ dev để chấm nếu chỉ học UIT-ViCTSD. Khi dùng nhãn dev, mỗi dòng nhận dự đoán ngoài fold từ 5 phần chia phân tầng với seed 42. Chọn `C` theo PR-AUC và chọn ngưỡng F1 trên các dự đoán ngoài fold bên trong tập huấn luyện. Với dữ liệu gộp, phần UIT-ViCTSD nằm trong các tập huấn luyện còn tập xác thực bên trong lấy từ dev. Trong cấu hình có bình luận cha, bỏ ngữ cảnh huấn luyện nếu nó trùng văn bản của một bình luận đang được chấm. Với mô hình chỉ học UIT-ViCTSD, cả `C` và ngưỡng F1 được chọn trên 5 fold của UIT-ViCTSD.

Ví dụ few-shot loại mọi văn bản xuất hiện trong dev hoặc corpus. Độ đo luôn là F1 của lớp `1` và PR-AUC (average precision). Khoảng tin cậy 95% lấy bằng bootstrap 1.000 lần với seed 42 trên từng nhóm.

Các notebook hỗ trợ Kaggle T4. Kết quả ghi ở đây chạy trên NVIDIA RTX A5000 cục bộ với Python 3.10.20, PyTorch 2.14.0 và CUDA 12.6 vì tài khoản Kaggle chưa xác thực được. Dùng Transformers 4.57.1 cho cả hai notebook, PyVi 0.1.1 cho PhoBERT, và bitsandbytes 0.48.2 với batch 16 cho Qwen. Các lần chạy CPU dùng Python 3.9.23 và scikit-learn 1.6.1.

## 4. Kết quả

Điểm mốc trước đối chiếu nhãn, trên cả 1.100 mẫu. CPU Intel Xeon Silver 4310, Python 3.9.23 và scikit-learn 1.6.1. Lần chạy TF-IDF tốn 46,4 giây cho huấn luyện và dự đoán.

| Model | F1 [KTC 95%] | PR-AUC [KTC 95%] |
|---|---:|---:|
| Đa số | 0,000 [0,000–0,000] | 0,155 [0,133–0,177] |
| Từ khóa | 0,057 [0,012–0,105] | 0,179 [0,149–0,211] |
| TF-IDF UIT-ViCTSD, ngưỡng 0,5 | 0,255 [0,195–0,310] | 0,236 [0,193–0,291] |

Điểm của cùng cấu hình TF-IDF UIT-ViCTSD sau lượt 1 đến lượt 5 lần lượt là F1 **0,272 / 0,289 / 0,293 / 0,305 / 0,314** và PR-AUC **0,260 / 0,277 / 0,279 / 0,293 / 0,302**. Mô hình này cũng xếp hạng mẫu để rà soát, nên diễn biến điểm phản ánh một phần cách chọn mẫu sửa nhãn.

PhoBERT chạy với seed 42, 43 và 44 trong lần lượt 82,71, 81,40 và 79,74 giây. Trung bình và độ lệch chuẩn giữa ba seed trên toàn dev là F1 **0,388 ± 0,005** và PR-AUC **0,379 ± 0,012**. Điểm của cấu hình ensemble trong bảng kết quả tính từ xác suất trung bình của ba seed.

### So sánh đủ 18 cấu hình

Tất cả điểm dưới đây dùng dev sau lượt 5, commit `3db6306`. Mỗi số đo ghi kèm khoảng tin cậy 95% trong ngoặc vuông. Thời gian TF-IDF là thời gian của một lần chạy cho mỗi cặp nguồn và đầu vào, dùng chung cho hai ngưỡng. Thời gian Qwen tính riêng từng prompt, sau khi nạp model.

| Cấu hình | Giây | F1 toàn bộ | PR-AUC toàn bộ | F1 gốc | PR-AUC gốc | F1 trả lời | PR-AUC trả lời |
|---|---:|---:|---:|---:|---:|---:|---:|
| `majority` | 0.00 | 0.000 [0.000–0.000] | 0.168 [0.146–0.192] | 0.000 [0.000–0.000] | 0.081 [0.054–0.111] | 0.000 [0.000–0.000] | 0.219 [0.189–0.250] |
| `keywords` | 0.03 | 0.053 [0.011–0.097] | 0.191 [0.162–0.221] | 0.000 [0.000–0.000] | 0.081 [0.054–0.111] | 0.064 [0.013–0.118] | 0.245 [0.205–0.280] |
| `tfidf_lr_victsd_comment_t05` | 47.50 | 0.314 [0.257–0.374] | 0.302 [0.245–0.373] | 0.385 [0.270–0.482] | 0.412 [0.264–0.588] | 0.281 [0.213–0.346] | 0.320 [0.260–0.391] |
| `tfidf_lr_victsd_comment_tf1` | 47.50 | 0.297 [0.233–0.355] | 0.302 [0.245–0.373] | 0.375 [0.256–0.492] | 0.412 [0.264–0.588] | 0.261 [0.193–0.332] | 0.320 [0.260–0.391] |
| `tfidf_lr_victsd_parent_t05` | 57.60 | 0.282 [0.218–0.345] | 0.281 [0.226–0.348] | 0.385 [0.270–0.482] | 0.412 [0.264–0.588] | 0.219 [0.150–0.297] | 0.309 [0.248–0.376] |
| `tfidf_lr_victsd_parent_tf1` | 57.60 | 0.268 [0.198–0.330] | 0.281 [0.226–0.348] | 0.375 [0.256–0.492] | 0.412 [0.264–0.588] | 0.206 [0.135–0.287] | 0.309 [0.248–0.376] |
| `tfidf_lr_dev_comment_t05` | 14.41 | 0.482 [0.418–0.542] | 0.476 [0.407–0.553] | 0.338 [0.194–0.464] | 0.276 [0.178–0.427] | 0.517 [0.446–0.582] | 0.547 [0.465–0.633] |
| `tfidf_lr_dev_comment_tf1` | 14.41 | 0.505 [0.455–0.557] | 0.476 [0.407–0.553] | 0.324 [0.204–0.436] | 0.276 [0.178–0.427] | 0.551 [0.497–0.608] | 0.547 [0.465–0.633] |
| `tfidf_lr_dev_parent_t05` | 48.83 | 0.402 [0.341–0.466] | 0.425 [0.359–0.505] | 0.241 [0.087–0.378] | 0.262 [0.166–0.419] | 0.434 [0.364–0.505] | 0.477 [0.394–0.560] |
| `tfidf_lr_dev_parent_tf1` | 48.83 | 0.492 [0.440–0.545] | 0.425 [0.359–0.505] | 0.326 [0.191–0.454] | 0.262 [0.166–0.419] | 0.526 [0.467–0.578] | 0.477 [0.394–0.560] |
| `tfidf_lr_victsd_plus_dev_comment_t05` | 305.55 | 0.468 [0.415–0.515] | 0.424 [0.356–0.505] | 0.355 [0.248–0.453] | 0.457 [0.294–0.633] | 0.507 [0.443–0.566] | 0.448 [0.367–0.534] |
| `tfidf_lr_victsd_plus_dev_comment_tf1` | 305.55 | 0.461 [0.411–0.508] | 0.424 [0.356–0.505] | 0.329 [0.222–0.428] | 0.457 [0.294–0.633] | 0.505 [0.444–0.567] | 0.448 [0.367–0.534] |
| `tfidf_lr_victsd_plus_dev_parent_t05` | 353.19 | 0.390 [0.347–0.433] | 0.380 [0.315–0.462] | 0.362 [0.256–0.459] | 0.383 [0.247–0.578] | 0.396 [0.345–0.442] | 0.393 [0.318–0.477] |
| `tfidf_lr_victsd_plus_dev_parent_tf1` | 353.19 | 0.416 [0.358–0.470] | 0.380 [0.315–0.462] | 0.392 [0.258–0.512] | 0.383 [0.247–0.578] | 0.422 [0.353–0.482] | 0.393 [0.318–0.477] |
| `phobert_victsd_comment` | 251.31 | 0.385 [0.322–0.444] | 0.381 [0.314–0.461] | 0.407 [0.289–0.517] | 0.469 [0.314–0.679] | 0.376 [0.305–0.444] | 0.409 [0.335–0.490] |
| `qwen_zero_parent` | 108.54 | 0.470 [0.409–0.530] | 0.452 [0.382–0.532] | 0.494 [0.346–0.620] | 0.588 [0.415–0.743] | 0.463 [0.382–0.534] | 0.438 [0.365–0.523] |
| `qwen_few_parent` | 213.28 | 0.429 [0.387–0.475] | 0.401 [0.340–0.474] | 0.309 [0.220–0.396] | 0.480 [0.314–0.648] | 0.465 [0.413–0.512] | 0.392 [0.328–0.470] |
| `qwen_cot_parent` | 855.13 | 0.501 [0.445–0.551] | 0.322 [0.273–0.381] | 0.391 [0.279–0.489] | 0.282 [0.171–0.415] | 0.535 [0.474–0.585] | 0.359 [0.306–0.426] |

Trong ba prompt Qwen, CoT đạt F1 cao nhất (0,501) và zero-shot đạt PR-AUC cao nhất (0,452). Few-shot tăng recall lên 0,908 và gán `1` cho 54,4% dev, nhưng precision giảm còn 0,281.

### Ablation trên dev sau lượt 5

- **Nguồn huấn luyện.** Với văn bản bình luận và ngưỡng 0,5, F1 đạt 0,314 trên UIT-ViCTSD, 0,482 khi học dev, và 0,468 khi gộp hai nguồn. PR-AUC tương ứng là 0,302, 0,476 và 0,424. UIT-ViCTSD có tín hiệu hữu ích cho bình luận gốc, với PR-AUC 0,457 ở cấu hình gộp và 0,276 ở cấu hình chỉ dev.
- **Bình luận cha.** Thêm văn bản cha giảm PR-AUC ở cả ba cách chọn nguồn. Với dev là nguồn huấn luyện, F1 giảm từ 0,482 xuống 0,402 ở ngưỡng 0,5 và từ 0,505 xuống 0,492 khi dùng ngưỡng F1. Kết quả áp dụng cho cách biểu diễn n-gram của thí nghiệm này.
- **Ngưỡng.** Chọn ngưỡng từ fold huấn luyện tăng F1 từ 0,482 lên 0,505 với dev và văn bản bình luận. Với UIT-ViCTSD, ngưỡng huấn luyện 0,560 giảm F1 từ 0,314 xuống 0,297 trên dev.

## 5. Model cuối

Chọn `tfidf_lr_dev_comment_tf1`, tức TF-IDF và Logistic Regression cân bằng lớp học trên dev, dùng văn bản bình luận và ngưỡng chọn theo F1 trong phần huấn luyện. Cấu hình đạt F1 **0,505** [0,455–0,557] và PR-AUC **0,476** [0,407–0,553] ngoài fold. Precision đạt 0,394 và recall đạt 0,703. F1 của nhóm gốc là 0,324, còn nhóm trả lời là 0,551.

So với Qwen CoT, cấu hình này có F1 cao hơn 0,004. Bootstrap ghép cặp 1.000 lần cho khoảng tin cậy 95% [−0,047 đến 0,054] đối với chênh lệch F1. TF-IDF có PR-AUC cao hơn và chạy phép đánh giá cấu hình đã chọn trong 14,41 giây trên CPU, so với 855,13 giây suy luận CoT trên GPU.

Huấn luyện lại trên cả 1.100 dòng dev cho `C=1,0` và ngưỡng `0,4694277`. Lần huấn luyện lại và gán nhãn mất 10,47 giây. Model dự đoán 803/20.468 bình luận là `1` (3,92%). Sau khi giữ nhãn dev cho 1.103 dòng trùng nội dung, `corpus_labels.csv` có **751/20.468** nhãn `1` (**3,67%**). Tỷ lệ gán `1` là 2,05% ở bình luận gốc (266/12.977) và 6,47% ở câu trả lời (485/7.491). Đã kiểm tra đủ ID theo thứ tự corpus, đủ ba cột `id,label,score` và điểm model trong [0, 1].

## 6. Phân tích lỗi

Các ví dụ dưới đây lấy từ 150 bình luận đã rà soát.

- **Mốc.** Đoán lớp đa số bỏ sót mọi mẫu dương. Luật từ khóa chạy dưới một giây và dễ giải thích, nhưng bỏ sót `060`, trong đó người viết coi thường ý kiến qua câu hỏi tu từ và cách chất vấn người đối thoại.
- **TF-IDF.** Model học từ UIT-ViCTSD cho điểm cao ở `391` và `R140`, nơi từ mạnh đi cùng lời lên án một hành vi hoặc một lập luận có tình tiết. Model học dev xử lý tốt hơn nhiều lời coi thường theo cách viết VnExpress, nhưng vẫn nhầm câu nhắc nhẹ `R071` và bỏ sót lời mỉa mai “bắn rụng bằng niềm tin” ở `E131`. Cấu hình học dev chỉ cần 1.100 nhãn và chạy trên CPU.
- **PhoBERT.** Fine-tune trên dữ liệu công khai cho phép học cách dùng từ ngoài danh sách quy tắc. Model vẫn nhầm `R028` khi lời chê hướng vào phim và bỏ sót lối gạt ý kiến “mời bạn tìm hiểu lại” ở `R026`. Chi phí huấn luyện cao hơn TF-IDF và nhãn độc hại của UIT-ViCTSD vẫn ảnh hưởng kết quả.
- **Qwen zero-shot.** Prompt dùng được định nghĩa và bình luận cha mà không huấn luyện thêm. Mô hình vẫn có thể nhầm đối tượng bị nhắm tới. Ví dụ `E222` phản đối quy chụp người xăm hình nhưng nhận nhãn `1`. Ngược lại, `R248` dùng “xem phim nhiều quá” để gạt ý kiến và nhận nhãn `0`.
- **Qwen few-shot và CoT.** Hai prompt này nhận ra ý coi thường ở `R026` mà zero-shot bỏ sót. Cả ba prompt vẫn nhầm `391` vì từ mạnh mô tả hành vi giết trẻ em, và bỏ sót lời mỉa mai dựa vào ngữ cảnh ở `E131`. Few-shot chỉ cần tám ví dụ có nhãn. CoT cho F1 tốt hơn few-shot trong thí nghiệm này với thời gian suy luận khoảng bốn lần lâu hơn.

Điểm dùng làm căn cứ chọn model lấy từ tập dev lấy mẫu tăng dương. 950 nhãn ban đầu chưa được rà soát trong năm lượt và dự án chưa có tập test riêng.

## 7. Chạy lại

```bash
python -m src.classifiers.incivility initial
python -m src.classifiers.incivility review --round 1
python -m src.classifiers.incivility apply-review --round 1
```

`review` tạo phiếu trong `results/incivility/reviews/`. Điền cột `label` và ghi lý do vào cột `reason` nếu cần. Chạy `apply-review`, commit riêng cột nhãn đã sửa, rồi chạy lại `initial` trước khi tạo phiếu cho lượt tiếp theo. Để tái hiện bảng so sánh, dùng trực tiếp `dev.csv` đã commit.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.classifiers.incivility ladder --jobs 4
python -m src.classifiers.incivility kaggle-bundle
```

Chạy `notebooks/incivility_phobert.ipynb` và `notebooks/incivility_qwen.ipynb` trên Kaggle GPU với Internet bật và bundle đính kèm dưới dạng dataset private. Có thể chạy cùng các cell trên GPU CUDA cục bộ từ thư mục gốc repo. Đặt `VISIM_INCIVILITY_QWEN_BATCH=16` để lặp lại batch đã dùng. Đưa các file `dev_scores_*.csv` và `*_dev.json` vào `results/incivility/kaggle/`, rồi chạy:

```bash
python -m src.classifiers.incivility collect
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.classifiers.incivility final --config tfidf_lr_dev_comment_tf1
```

Kết quả thô nằm trong `results/incivility/`. `ladder_full.csv` chứa toàn bộ điểm, `phobert_seed_metrics.csv` chứa điểm từng seed và `final.json` ghi cấu hình, ngưỡng và tỷ lệ dương của lần gán nhãn corpus.
