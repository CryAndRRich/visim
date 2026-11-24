# Thiếu văn minh (Tấn)

## 1. Dữ liệu

- `dev.csv`: 1.100 bình luận VnExpress. Trước khi rà soát có 930 nhãn `0` và 170 nhãn `1`. Sau lượt 5 có 915 nhãn `0` và 185 nhãn `1`. Phân theo `parent_text`, có 407 bình luận gốc với 33 nhãn `1` và 693 bình luận trả lời với 152 nhãn `1`.
- Dev gồm ba tầng, nhận biết qua tiền tố `id`:
  - `001` đến `500`: mẫu ngẫu nhiên từ corpus, 28 nhãn `1` (5,6%), tỷ lệ câu trả lời 35,4% (corpus: 36,6%). Đây là phần duy nhất đại diện cho corpus.
  - `E…` (300 mẫu, 35 nhãn `1`, toàn câu trả lời) và `R…` (300 mẫu, 122 nhãn `1`): hai tầng chọn thêm để tăng số mẫu dương.
  - Vì hai tầng này, tỷ lệ dương của dev (16,8%) cao gấp ba tầng ngẫu nhiên. Tỷ lệ dương của dev chỉ dùng để mô tả mẫu đánh giá, không dùng để ước lượng tỷ lệ trong corpus.
- `train.csv`: 10.000 bình luận UIT-ViCTSD với nhãn **toxic** làm nhãn thay thế cho thiếu văn minh. Loại hai bình luận trùng văn bản với mục cần chấm trong dev sau khi chuẩn hóa Unicode, hoa thường và khoảng trắng. Còn 9.998 mẫu, trong đó 1.100 mẫu dương. Hai bộ dữ liệu dùng định nghĩa khác nhau, nhất là với các câu thể hiện thái độ coi thường mà không dùng từ tục.
- `corpus.csv`: 20.468 bình luận, gồm 12.977 bình luận gốc và 7.491 câu trả lời. ID của dev dùng bí danh khác với ID của corpus. Ghép theo cặp nguyên văn `(text, parent_text)` (`common.dev_to_corpus_ids`) thì có 2 mẫu dev khớp với nhiều bình luận corpus giống hệt nhau. Mọi bản giống hệt đều nhận nhãn dev, tổng cộng 1.103 dòng corpus nhận nhãn từ dev. Cột `score` vẫn lấy từ model.

## 2. Các lượt rà soát nhãn

Nhãn dev ban đầu do LLM (Claude) gán theo định nghĩa thiếu văn minh trong `TASK.md`. Cả năm lượt rà soát dưới đây cũng do LLM (Claude) quyết định theo yêu cầu của Tấn, không có người gán nhãn độc lập. Vì vậy nhãn dev là nhãn LLM đã được LLM rà lại, và mọi điểm trong file này đo mức khớp với nhãn LLM. Mức khớp với người đọc thật được đo riêng ở mục 8.

Mỗi lượt dùng một phiếu ẩn nhãn và điểm cũ gồm 30 mẫu: 10 mẫu điểm cao nhất, 10 mẫu điểm thấp nhất và 10 mẫu gần ngưỡng lấy mẫu theo model TF-IDF học UIT-ViCTSD. Các lượt loại toàn bộ ID đã xem trước đó. Nhãn được sửa trực tiếp trong `dev.csv`; mỗi lượt là một commit, xem bằng git. Công cụ tạo phiếu đã được bỏ khỏi repo, bảng dưới đây là bản ghi lịch sử.

| Lượt | Cao / thấp / phân vân | 0 → 1 | 1 → 0 | Commit |
|---|---:|---:|---:|---|
| 1 | 10 / 10 / 10 | 6 | 2 | `48b0fdf` |
| 2 | 10 / 10 / 10 | 4 | 0 | `44bc371` |
| 3 | 10 / 10 / 10 | 2 | 1 | `1d4a091` |
| 4 | 10 / 10 / 10 | 4 | 0 | `2932250` |
| 5 | 10 / 10 / 10 | 2 | 0 | `3db6306` |

- Lượt 1: `R222` và `E205` đổi thành `1` vì có từ hạ thấp người phạm tội. `R140` đổi thành `0` vì đưa tình tiết để phản biện. `391` đổi thành `0` vì lời lên án hướng vào hành vi giết trẻ em. `060` kết hợp lý lẽ với câu hỏi tu từ coi thường người đưa ý kiến nên gán `1`.
- Lượt 2: `R010` và `R273` đổi thành `1` vì có câu giễu cợt hướng vào một người hoặc một nhóm. `431` gán `1` vì mỉa mai nhóm tài xế. `023` gán `0` vì không xác định được người bị nhắm tới.
- Lượt 3: `153` và `R099` đổi thành `1` vì người viết dùng giọng coi thường ý kiến hoặc năng lực của người phê bình. `R263` đổi thành `0` vì trọng tâm là cảnh báo nguy cơ bạo lực trong một số mối quan hệ.
- Lượt 4: `E131` chế giễu nhận định của người đối thoại. `E184` quy chụp cả một nhóm người dựa trên tình trạng sức khỏe. Cả hai đổi từ `0` sang `1`. `190` dùng một lời chê đứng riêng, chưa xác định được người bị công kích nên giữ nhãn `0`.
- Lượt 5: `E055` quy kết người gây án vốn có bản chất bạo lực. `R230` gán cho nhóm người vi phạm đặc điểm thiếu hiểu biết. Cả hai đổi từ `0` sang `1`.

Dừng sau lượt 5 theo tiêu chí tối đa khoảng hai nhãn sửa trong một lượt 30 mẫu. Tổng cộng đã xem 150 ID khác nhau và sửa 21 nhãn (18 từ `0` sang `1`, ba theo chiều ngược lại). 950 mẫu ngoài các lượt này giữ nhãn Claude ban đầu. Git diff cho thấy nội dung, thứ tự và định dạng CSV được giữ nguyên qua năm lượt.

## 3. Model và thiết kế so sánh

| Cấu hình | Dữ liệu và đầu vào | Thiết lập |
|---|---|---|
| Đa số | Tỷ lệ từ UIT-ViCTSD | Dự đoán lớp `0`, điểm cố định bằng tỷ lệ dương trong train |
| Từ khóa | Bình luận | Danh sách từ miệt thị, từ tục và cụm coi thường. Điểm quy tắc `1 − 2^(-số lần khớp)` |
| TF-IDF + Logistic Regression | UIT-ViCTSD, dev, hoặc cả hai. Thêm hoặc bỏ bình luận cha | N-gram từ 1–2 và ký tự 2–5, `min_df=2`, `sublinear_tf=True`. `class_weight=balanced`, `liblinear`, `C` trong 0,25 / 1 / 4 |
| PhoBERT | UIT-ViCTSD, văn bản bình luận tách từ bằng PyVi | Base v2. AdamW, LR 2e-5, batch 32, 3 epoch, tối đa 192 token, trọng số lớp nghịch tần suất. Lấy trung bình xác suất từ ba lần chạy với seed 42, 43, 44. [Phiên bản PhoBERT](https://huggingface.co/vinai/phobert-base-v2/commit/86cd7fd4c148980922ac11a2cf5e257f2ba639e1) |
| Qwen2.5 7B Instruct | Bình luận và bình luận cha; định nghĩa thiếu văn minh trong `TASK.md` | NF4 với tính toán FP16. Zero-shot, 8 ví dụ từ UIT-ViCTSD (4 mỗi lớp), và cùng 8 ví dụ với hướng dẫn suy luận. Seed 42, giải mã tham lam, tối đa 64 token suy luận. Lấy xác suất token `1` chuẩn hóa trên hai token nhãn. [Phiên bản Qwen](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/commit/a09a35458c702b33eeacc393d103063234e8bc28) |

TF-IDF dùng toàn bộ dev để chấm nếu chỉ học UIT-ViCTSD. Khi dùng nhãn dev, mỗi dòng nhận dự đoán ngoài fold từ 5 phần chia phân tầng với seed 42. Chọn `C` theo PR-AUC và chọn ngưỡng F1 (`common.f1_threshold`) trên các dự đoán ngoài fold bên trong tập huấn luyện. Với dữ liệu gộp, phần UIT-ViCTSD nằm trong các tập huấn luyện còn tập xác thực bên trong lấy từ dev. Trong cấu hình có bình luận cha, bỏ ngữ cảnh huấn luyện nếu nó trùng văn bản của một bình luận đang được chấm. Với mô hình chỉ học UIT-ViCTSD, cả `C` và ngưỡng F1 được chọn trên 5 fold của UIT-ViCTSD.

Ví dụ few-shot loại mọi văn bản xuất hiện trong dev hoặc corpus. Độ đo luôn là F1 của lớp `1` và PR-AUC (average precision). Khoảng tin cậy 95% lấy bằng bootstrap 1.000 lần với seed 42 trên từng nhóm (toàn bộ, gốc, trả lời, tầng ngẫu nhiên, tầng `E`, tầng `R`). Mọi điểm ghi ra CSV làm tròn 4 chữ số.

Các notebook hỗ trợ Kaggle T4. Kết quả GPU ghi ở đây chạy trên NVIDIA RTX A5000 cục bộ với Python 3.10.20, PyTorch 2.14.0 và CUDA 12.6 vì tài khoản Kaggle chưa xác thực được. Dùng Transformers 4.57.1 cho cả hai notebook, PyVi 0.1.1 cho PhoBERT, và bitsandbytes 0.48.2 với batch 16 cho Qwen. Các lần chạy CPU dùng Python 3.9 và scikit-learn 1.6.1.

## 4. Kết quả

PhoBERT chạy với seed 42, 43 và 44 trong lần lượt 82,71, 81,40 và 79,74 giây. Trung bình và độ lệch chuẩn giữa ba seed trên toàn dev là F1 **0,388 ± 0,005** và PR-AUC **0,379 ± 0,012** (`phobert_seed_summary.csv` do `collect` ghi). Điểm của cấu hình ensemble trong bảng kết quả tính từ xác suất trung bình của ba seed.

### So sánh đủ 18 cấu hình

Tất cả điểm dưới đây dùng dev sau lượt 5, commit `3db6306`. Mỗi số đo ghi kèm khoảng tin cậy 95% trong ngoặc vuông. Điểm TF-IDF và mốc do `ladder` tạo, điểm PhoBERT và Qwen do hai notebook tạo và `collect` chấm. Cột giây là số giây do lệnh ghi lại và phụ thuộc phần cứng: thời gian TF-IDF đo trên CPU Intel Xeon Silver 4310, là thời gian của một lần chạy cho mỗi cặp nguồn và đầu vào, dùng chung cho hai ngưỡng. Thời gian Qwen tính riêng từng prompt, sau khi nạp model. Điểm PhoBERT và Qwen tính từ file điểm trước khi notebook làm tròn 4 chữ số; chạy lại có thể lệch ở chữ số thứ ba (như hai dòng `victsd_plus_dev_comment` sau khi làm tròn).

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
| `tfidf_lr_victsd_plus_dev_comment_t05` | 305.55 | 0.468 [0.415–0.515] | 0.424 [0.356–0.506] | 0.355 [0.248–0.453] | 0.457 [0.294–0.633] | 0.507 [0.443–0.566] | 0.449 [0.367–0.535] |
| `tfidf_lr_victsd_plus_dev_comment_tf1` | 305.55 | 0.461 [0.411–0.508] | 0.424 [0.356–0.506] | 0.329 [0.222–0.428] | 0.457 [0.294–0.633] | 0.505 [0.444–0.567] | 0.449 [0.367–0.535] |
| `tfidf_lr_victsd_plus_dev_parent_t05` | 353.19 | 0.390 [0.347–0.433] | 0.380 [0.315–0.462] | 0.362 [0.256–0.459] | 0.383 [0.247–0.578] | 0.396 [0.345–0.442] | 0.393 [0.318–0.477] |
| `tfidf_lr_victsd_plus_dev_parent_tf1` | 353.19 | 0.416 [0.358–0.470] | 0.380 [0.315–0.462] | 0.392 [0.258–0.512] | 0.383 [0.247–0.578] | 0.422 [0.353–0.482] | 0.393 [0.318–0.477] |
| `phobert_victsd_comment` | 251.31 | 0.385 [0.322–0.444] | 0.381 [0.314–0.461] | 0.407 [0.289–0.517] | 0.469 [0.314–0.679] | 0.376 [0.305–0.444] | 0.409 [0.335–0.490] |
| `qwen_zero_parent` | 108.54 | 0.470 [0.409–0.530] | 0.452 [0.382–0.532] | 0.494 [0.346–0.620] | 0.588 [0.415–0.743] | 0.463 [0.382–0.534] | 0.438 [0.365–0.523] |
| `qwen_few_parent` | 213.28 | 0.429 [0.387–0.475] | 0.401 [0.340–0.474] | 0.309 [0.220–0.396] | 0.480 [0.314–0.648] | 0.465 [0.413–0.512] | 0.392 [0.328–0.470] |
| `qwen_cot_parent` | 855.13 | 0.501 [0.445–0.551] | 0.322 [0.273–0.381] | 0.391 [0.279–0.489] | 0.282 [0.171–0.415] | 0.535 [0.474–0.585] | 0.359 [0.306–0.426] |

Trong ba prompt Qwen, CoT đạt F1 cao nhất (0,501) và zero-shot đạt PR-AUC cao nhất (0,452). Few-shot tăng recall lên 0,908 và gán `1` cho 54,4% dev, nhưng precision giảm còn 0,281.

### Điểm theo tầng của dev

Điểm "toàn bộ" ở bảng trên gộp tầng ngẫu nhiên với hai tầng `E`, `R` được chọn thêm để tăng mẫu dương, nên cao hơn mức model đạt được trên corpus. Bảng dưới lấy từ `ladder.csv` (nhóm `random`, `stratum_E`, `stratum_R`) cho một số cấu hình TF-IDF. Với PhoBERT và Qwen, `collect` ghi các nhóm này vào `ladder_full.csv`; chưa tính được trong lần dọn mã này vì file kết quả notebook không có trên máy.

| Cấu hình | F1 ngẫu nhiên | PR-AUC ngẫu nhiên | F1 `E` | PR-AUC `E` | F1 `R` | PR-AUC `R` |
|---|---:|---:|---:|---:|---:|---:|
| `tfidf_lr_victsd_comment_t05` | 0.218 [0.126–0.317] | 0.207 [0.106–0.382] | 0.216 [0.090–0.344] | 0.265 [0.159–0.420] | 0.400 [0.318–0.477] | 0.465 [0.392–0.563] |
| `tfidf_lr_dev_comment_t05` | 0.093 [0.000–0.213] | 0.136 [0.076–0.269] | 0.314 [0.148–0.478] | 0.405 [0.258–0.576] | 0.566 [0.502–0.630] | 0.577 [0.490–0.666] |
| `tfidf_lr_dev_comment_tf1` | 0.216 [0.091–0.342] | 0.136 [0.076–0.269] | 0.442 [0.286–0.579] | 0.405 [0.258–0.576] | 0.577 [0.515–0.635] | 0.577 [0.490–0.666] |
| `tfidf_lr_victsd_plus_dev_comment_tf1` | 0.259 [0.162–0.356] | 0.231 [0.121–0.386] | 0.302 [0.186–0.407] | 0.318 [0.204–0.482] | 0.601 [0.536–0.667] | 0.534 [0.451–0.628] |

Trên tầng ngẫu nhiên (500 mẫu, chỉ 28 mẫu dương), mốc PR-AUC của đoán ngẫu nhiên là 0,056. Mọi khoảng tin cậy ở tầng này rất rộng, và thứ tự giữa các cấu hình khác với thứ tự trên toàn dev: các cấu hình có UIT-ViCTSD xếp hạng tốt hơn ở đây.

### Ablation trên dev sau lượt 5

- **Nguồn huấn luyện.** Với văn bản bình luận và ngưỡng 0,5, F1 đạt 0,314 trên UIT-ViCTSD, 0,482 khi học dev, và 0,468 khi gộp hai nguồn. PR-AUC tương ứng là 0,302, 0,476 và 0,424. UIT-ViCTSD có tín hiệu hữu ích cho bình luận gốc, với PR-AUC 0,457 ở cấu hình gộp và 0,276 ở cấu hình chỉ dev.
- **Bình luận cha.** Thêm văn bản cha giảm PR-AUC ở cả ba cách chọn nguồn. Với dev là nguồn huấn luyện, F1 giảm từ 0,482 xuống 0,402 ở ngưỡng 0,5 và từ 0,505 xuống 0,492 khi dùng ngưỡng F1. Kết quả áp dụng cho cách biểu diễn n-gram của thí nghiệm này.
- **Ngưỡng.** Chọn ngưỡng từ fold huấn luyện tăng F1 từ 0,482 lên 0,505 với dev và văn bản bình luận. Với UIT-ViCTSD, ngưỡng huấn luyện 0,560 giảm F1 từ 0,314 xuống 0,297 trên dev.

## 5. Model cuối

Chọn `tfidf_lr_dev_comment_tf1`, tức TF-IDF và Logistic Regression cân bằng lớp học trên dev, dùng văn bản bình luận và ngưỡng chọn theo F1 trong phần huấn luyện. Cấu hình đạt F1 **0,505** [0,455–0,557] và PR-AUC **0,476** [0,407–0,553] ngoài fold trên toàn dev. Precision đạt 0,394 và recall đạt 0,703. F1 của nhóm gốc là 0,324, còn nhóm trả lời là 0,551.

**Điểm toàn bộ dev bị thổi phồng.** F1 0,505 và PR-AUC 0,476 tính trên cả dev, trong đó hai tầng chọn thêm chiếm 600/1.100 mẫu và 157/185 mẫu dương. Trên tầng `E` cấu hình đạt F1 0,442 và PR-AUC 0,405, trên tầng `R` đạt 0,577 và 0,577. Trên tầng ngẫu nhiên, phần duy nhất đại diện cho corpus, cấu hình chỉ đạt F1 **0,216** [0,091–0,342] và PR-AUC **0,136** [0,076–0,269] (mốc 0,056). Đây mới là con số nên dùng khi nói về chất lượng nhãn corpus.

Model được chọn trong 18 cấu hình trên chính tập dev dùng để báo điểm, nên điểm của model được chọn có thể hơi lạc quan. So với Qwen CoT, cấu hình này có F1 cao hơn 0,004 với khoảng tin cậy 95% của chênh lệch [−0,047 đến 0,054] (bootstrap ghép cặp 1.000 lần, `compare tfidf_lr_dev_comment_tf1 qwen_cot_parent`), tức hai model không khác nhau rõ về F1. Số này cần file `dev_scores_qwen_cot_parent.csv` của notebook trong `results/incivility/kaggle/`; file đó không có trên máy chạy lần dọn mã này nên chưa chạy lại được sau khi điểm được làm tròn 4 chữ số. Trên toàn dev TF-IDF có PR-AUC cao hơn và chạy phép đánh giá cấu hình đã chọn trong 14,41 giây trên CPU, so với 855,13 giây suy luận CoT trên GPU.

Huấn luyện lại trên cả 1.100 dòng dev cho `C=1,0` và ngưỡng `0,4694`. Model dự đoán 803/20.468 bình luận là `1` (3,92%). Sau khi giữ nhãn dev cho 1.103 dòng trùng nội dung, `corpus_labels.csv` có **751/20.468** nhãn `1` (**3,67%**). Tỷ lệ gán `1` là 2,05% ở bình luận gốc (266/12.977) và 6,47% ở câu trả lời (485/7.491). Các số này do `final` in ra và ghi vào `results/incivility/final.json`. File có đúng ba cột `id,label,score`, đủ ID theo thứ tự corpus, điểm model làm tròn 4 chữ số.

**Lưu ý về ngưỡng.** Ngưỡng 0,4694 tối ưu F1 trên dự đoán ngoài fold của dev, nơi tỷ lệ dương là 16,8%, với `class_weight="balanced"`. Trên các bình luận ngoài dev, model chỉ gán `1` cho 2,92%, trong khi tầng ngẫu nhiên của dev có 5,6% nhãn `1`. Như vậy nhãn corpus nhiều khả năng đếm thiếu bình luận thiếu văn minh (khoảng một nửa so với ước lượng từ tầng ngẫu nhiên). Model và ngưỡng được giữ nguyên; khi dùng nhãn corpus để so sánh tỷ lệ giữa các nhóm bài, nên xem đây là ước lượng thấp và ưu tiên so sánh tương đối hoặc dùng cột `score`.

## 6. Phân tích lỗi

Các ví dụ dưới đây lấy từ 150 bình luận đã rà soát.

- **Mốc.** Đoán lớp đa số bỏ sót mọi mẫu dương. Luật từ khóa chạy dưới một giây và dễ giải thích, nhưng bỏ sót `060`, trong đó người viết coi thường ý kiến qua câu hỏi tu từ và cách chất vấn người đối thoại.
- **TF-IDF.** Model học từ UIT-ViCTSD cho điểm cao ở `391` và `R140`, nơi từ mạnh đi cùng lời lên án một hành vi hoặc một lập luận có tình tiết. Model học dev xử lý tốt hơn nhiều lời coi thường theo cách viết VnExpress, nhưng vẫn nhầm câu nhắc nhẹ `R071` và bỏ sót lời mỉa mai dựa vào ngữ cảnh ở `E131`. Cấu hình học dev chỉ cần 1.100 nhãn và chạy trên CPU.
- **PhoBERT.** Fine-tune trên dữ liệu công khai cho phép học cách dùng từ ngoài danh sách quy tắc. Model vẫn nhầm `R028` khi lời chê hướng vào phim và bỏ sót lối gạt ý kiến người đối thoại ở `R026`. Chi phí huấn luyện cao hơn TF-IDF và nhãn độc hại của UIT-ViCTSD vẫn ảnh hưởng kết quả.
- **Qwen zero-shot.** Prompt dùng được định nghĩa và bình luận cha mà không huấn luyện thêm. Mô hình vẫn có thể nhầm đối tượng bị nhắm tới. Ví dụ `E222` phản đối quy chụp người xăm hình nhưng nhận nhãn `1`. Ngược lại, `R248` gạt ý kiến người khác bằng một lời chê bóng gió và nhận nhãn `0`.
- **Qwen few-shot và CoT.** Hai prompt này nhận ra ý coi thường ở `R026` mà zero-shot bỏ sót. Cả ba prompt vẫn nhầm `391` vì từ mạnh mô tả hành vi giết trẻ em, và bỏ sót lời mỉa mai dựa vào ngữ cảnh ở `E131`. Few-shot chỉ cần tám ví dụ có nhãn. CoT cho F1 tốt hơn few-shot trong thí nghiệm này với thời gian suy luận khoảng bốn lần lâu hơn.

Hạn chế:
- Điểm dùng làm căn cứ chọn model lấy từ tập dev lấy mẫu tăng dương; điểm trên tầng ngẫu nhiên thấp hơn nhiều (mục 4). 950 nhãn ban đầu chưa được rà soát trong năm lượt và dự án chưa có tập test riêng.
- Qwen CoT chỉ được sinh tối đa 64 token suy luận (`MAX_REASON` trong `notebooks/incivility_qwen.ipynb`) trước khi đọc nhãn, nên lập luận dài có thể bị cắt. Chưa chạy lại với giới hạn lớn hơn.

## 7. Chạy lại

```bash
python -m src.classifiers.incivility initial
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.classifiers.incivility ladder --jobs 4
python -m src.classifiers.incivility kaggle-bundle
```

`initial` chấm nhanh các mốc và TF-IDF học UIT-ViCTSD. `ladder` chấm toàn bộ mốc và 12 cấu hình TF-IDF, ghi `results/incivility/ladder.csv`, `metrics.json` và `dev_scores_<cấu hình>.csv` (cột `id,pred,score`).

Chạy `notebooks/incivility_phobert.ipynb` và `notebooks/incivility_qwen.ipynb` trên Kaggle GPU với Internet bật và bundle đính kèm dưới dạng dataset private. Có thể chạy cùng các cell trên GPU CUDA cục bộ từ thư mục gốc repo. Đặt `VISIM_INCIVILITY_QWEN_BATCH=16` để lặp lại batch đã dùng. Đưa các file `dev_scores_*.csv` và `*_dev.json` vào `results/incivility/kaggle/` (`collect` cần cả file JSON đi kèm để kiểm tra dữ liệu đầu vào), rồi chạy:

```bash
python -m src.classifiers.incivility collect
python -m src.classifiers.incivility compare tfidf_lr_dev_comment_tf1 qwen_cot_parent
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.classifiers.incivility final --config tfidf_lr_dev_comment_tf1
```

Kết quả thô nằm trong `results/incivility/`. `ladder_full.csv` chứa toàn bộ điểm theo từng nhóm, `phobert_seed_metrics.csv` chứa điểm từng seed và `final.json` ghi cấu hình, ngưỡng và các tỷ lệ dương của lần gán nhãn corpus.

## 8. Độ đồng thuận với người

Chưa thực hiện. Vì nhãn dev do LLM gán và rà lại (mục 2), cần đo mức khớp với một người gán độc lập theo quy trình trong `TASK.md`:

1. Tấn chạy `python -m src.classifiers.agreement sample incivility` để tạo `data/tasks/incivility/agreement.csv`: 100 mẫu dev lấy ngẫu nhiên, phân tầng theo nhãn, seed 42, cột `label` để trống.
2. Kiên gán cột `label` bằng tay theo định nghĩa thiếu văn minh trong `TASK.md`, không xem `dev.csv` và không dùng công cụ AI.
3. Chạy `python -m src.classifiers.agreement score incivility` để tính tỷ lệ khớp và Cohen's kappa (ghi `results/incivility/agreement.json`), commit `agreement.csv` và chép số mẫu, tỷ lệ khớp, kappa và các `id` lệch vào mục này.

Không sửa `dev.csv` theo kết quả này. Nếu kappa dưới khoảng 0,6, báo cả nhóm trước khi dùng nhãn thiếu văn minh cho báo cáo.
