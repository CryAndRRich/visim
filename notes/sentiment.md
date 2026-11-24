# Cảm xúc (sentiment): ghi chép

> **Nhãn `dev` của bài này là nhãn LLM, đã được LLM rà lại, không phải nhãn người kiểm tra.** Nhãn ban đầu do Claude gán. Ba lượt rà soát (120 mẫu) cũng do Claude (`claude-opus-5-5`) quyết thay người. Cả ba bài toán của nhóm đều ở cùng tình trạng này (xem `TASK.md`, quy trình lặp). Vì vậy mọi điểm số dưới đây đo mức khớp với nhãn LLM, không phải với người đọc thật. Độ đồng thuận với một người gán độc lập được đo riêng ở mục 8.

Người phụ trách: Dũng. Branch: `sentiment-dung`. Code: `src/classifiers/sentiment.py`, `notebooks/sentiment_visobert.ipynb`, `notebooks/sentiment_qwen.ipynb`.

> Mỗi bảng điểm ghi kèm `dev_labels_sha` (12 ký tự đầu của SHA-256 trên cặp `id,label` của `dev.csv`, do `ladder`, `collect` và `compare` in ra), để biết điểm tính trên nhãn của lượt nào.

## 1. Dữ liệu

- `dev.csv`: 1.100 bình luận VnExpress. Nhãn ban đầu do LLM gán (commit `cefae3a`, `dev_labels_sha = 7e5f9493db91`): 518 `neg`, 432 `neu`, 150 `pos`.
  - 63% là bình luận trả lời (có `parent_text`), so với 36,6% trong `corpus.csv` (`dev_reply_share`, `corpus_reply_share` trong `results/sentiment/metrics.json`). Tập `dev` cũng đã được chọn thêm mẫu dương, nên không dùng để ước lượng tỷ lệ nhãn.
  - `id` của `dev` là mã riêng (`001`, `E288`, ...), không phải `comment_id`. Để nối với `corpus.csv`, ghép theo (`text`, `parent_text`) bằng `common.dev_to_corpus_ids`: cả 1.100 mẫu đều khớp, 2 mẫu khớp với hơn một bình luận giống hệt nhau (tổng 1.103 dòng corpus, `final` in ra).
- `corpus.csv`: 20.468 bình luận.
- Không có `train.csv`: model học từ `dev` được chấm ngoài fold (5 fold, `StratifiedKFold(shuffle=True, random_state=42)`, siêu tham số chọn bằng kiểm định chéo lồng trong phần huấn luyện).
- Phân bố nhãn sau 3 lượt rà soát (`dev_labels_sha = 7c078c6bb311`): 524 `neg`, 427 `neu`, 149 `pos`. Tổng cộng 7/1.100 nhãn khác ban đầu (`git diff cefae3a -- data/tasks/sentiment/dev.csv`).

## 2. Các lượt sửa nhãn

Mỗi lượt, model tốt nhất lúc đó chọn 40 mẫu `dev` chưa kiểm tra ở lượt trước: với mỗi lớp, 10 mẫu model tin chắc nhất thuộc lớp đó, và 10 mẫu có chênh lệch nhỏ nhất giữa hai lớp được đoán cao nhất. Claude gán lại các mẫu này theo định nghĩa nhãn trong `TASK.md`, rồi nhãn mới được sửa thẳng vào `dev.csv`. Ở lượt 1 và 2, người gán không thấy nhãn hiện tại và dự đoán của model. Lượt 3 không còn mù: quy ước mới "mẫu còn mơ hồ thì giữ nhãn hiện tại" (quy ước 4 bên dưới) cần biết nhãn hiện tại. Công cụ chọn mẫu và file của từng lượt đã bỏ khỏi repo; bảng dưới và lịch sử git của `dev.csv` là bản ghi còn lại.

| Lượt | Model chọn mẫu | Kiểm tra (chắc neg / neu / pos / phân vân) | Sửa (giữ lại sau lượt 3) | Chiều sửa (giữ lại) | Commit |
|---|---|---|---|---|---|
| 1 | `visobert_pre_text` (không học từ `dev`) | 10 / 10 / 10 / 10 | 3 (2) | neu thành neg (R123), pos thành neu (211) | `ec7c0dd` |
| 2 | `tfidf_lr_ctx_oof` (tốt nhất sau lượt 1) | 10 / 10 / 10 / 10 | 5 (3) | neu thành neg (R212, 267, 180) | `eb21099` |
| 3 | `tfidf_lr_ctx_oof` (tốt nhất sau lượt 2) | 10 / 10 / 10 / 10 | 2, và trả về 3 (2) | neu thành neg (001, 147) | `fbdbece` |

Ở lượt 3, theo quy ước 4, 3 thay đổi của lượt 1-2 được trả về nhãn cũ: 202 (`neg`), E089 (`pos`), R254 (`neg`). Kết quả cuối: 7 nhãn khác ban đầu, gồm 001, 147, 180, 267, R123, R212 (`neu` thành `neg`) và 211 (`pos` thành `neu`).

**Dừng sau lượt 3:** lượt này chỉ sửa 2/40 mẫu, đúng điều kiện dừng trong `TASK.md` (khoảng 1-2 trên 30-40). Điểm của model tốt nhất cũng không đổi đáng kể qua các lượt (0,564, 0,568, 0,570, 0,568; chạy `ladder` trên `dev.csv` của từng commit), đều nằm trong khoảng tin cậy.

**Kiểu lỗi hay gặp của nhãn ban đầu:** gần như mọi chỗ sửa đều theo chiều `neu` thành `neg`.
- Câu hỏi vặn hoặc mỉa mai, bề ngoài là câu hỏi hay câu ngắn nên bị gán `neu`: R123, 180, 267.
- Lời lo ngại nói nhẹ, hoặc khen trước rồi lo ngại sau: 001, 147.
- Lập luận rằng một việc là bất công: R212.

**Quy ước thêm khi định nghĩa nhãn chưa rõ** (áp dụng cho lượt 3 và khi rà lại):
1. Câu hỏi vặn, lời coi thường người khác (bảo người kia đọc lại bài, hỏi người kia đã hiểu chưa, thách người kia tự làm): `neg`. Phản biện có lý lẽ, đính chính số liệu mà không hạ thấp người kia: `neu` (ví dụ 307, E299, 132).
2. Lời mong, chúc điều tốt không kèm than phiền: `pos` (E288, 044, 154). Lời mong kèm buồn, cảm thông hay lo lắng: `neg` (008, 093, 102). Lời mong mang tính yêu cầu: `neu` (436).
3. Có ý khen trước rồi ý chính sau "thế nhưng" là lo, chê: `neg` (147, 275).
4. Mẫu còn mơ hồ sau khi đọc kỹ thì giữ nhãn hiện tại. Ví dụ E240 (câu trấn an có thể đọc thành mỉa mai), 202, E089.

**Rà lại cùng kiểu (sau lượt 3):** tìm trong các mẫu `dev` chưa kiểm tra những mẫu cùng kiểu với quy ước 1 (coi thường, bảo người kia đọc bài) và quy ước 2 (mở đầu bằng lời mong, hy vọng, chúc). Các mẫu tìm được đều đã khớp quy ước, không sửa thêm.

## 3. Các model

| Mức | Cấu hình | Đầu vào | Học từ | Siêu tham số | Chạy ở đâu |
|---|---|---|---|---|---|
| Mốc | `majority` | không | đếm lớp trên `dev` | không | máy cá nhân |
| Mốc | `sopmi_oof` | `text` | từ điển SO-PMI trên `corpus.csv` (không nhãn); 2 ngưỡng chọn trên 4/5 `dev` | 14 từ hạt giống mỗi cực, âm tiết và bigram âm tiết, df ít nhất 5 | máy cá nhân |
| Cổ điển | `tfidf_lr_text_oof` | `text` | `dev` (ngoài fold) | TF-IDF từ 1-2 gram và ký tự 2-5 gram; LR, `C` trong {0,3; 1; 3; 10}, `class_weight` trong {None, balanced}, chọn bằng CV lồng | máy cá nhân |
| Cổ điển | `tfidf_lr_ctx_oof` | `text` + `parent_text` | `dev` (ngoài fold) | như trên, thêm khối TF-IDF riêng cho `parent_text` | máy cá nhân |
| Học sâu | `visobert_pre_text` | `text` | không huấn luyện | `5CD-AI/Vietnamese-Sentiment-visobert`; đầu ra theo thứ tự `NEG, POS, NEU` (`config.id2label`), đã đổi về `neg, neu, pos` | máy cá nhân, CPU |
| Học sâu | `visobert_pre_ctx` | cặp (`text`, `parent_text`) | không huấn luyện | như trên | máy cá nhân, CPU |
| Học sâu | `visobert_ft_{text,ctx}_oof` | như trên | `dev` (ngoài fold) | 4 epoch, lr 2e-5, batch 16, max_len 256, 3 seed (42, 43, 44); cố định trước, không chỉnh trên `dev` | Kaggle |
| LLM | `qwen7b_{zero,few,cot}` | `text` | không huấn luyện; few-shot lấy 3 ví dụ mỗi lớp từ 4 fold còn lại | Qwen2.5-7B-Instruct 4-bit nf4, nhãn là chữ số 0/1/2, điểm là softmax trên 3 token | Kaggle |
| LLM | `qwen7b_few_ctx` | `text` + `parent_text` | như trên | như trên | Kaggle |

Thời gian chạy `ladder` (cả 6 cấu hình trên máy): khoảng 1,5 phút trên máy của Dũng (macOS, 11 lõi CPU, Python 3.9, torch 2.8); khoảng 27 phút đo lại ngày 25/11/2026 trên MacBook Intel i3 (4 luồng, RAM 8 GB, Python 3.9, torch 2.2), trong lúc máy đang chạy song song các việc khác; phần lớn thời gian là ViSoBERT dùng sẵn trên CPU. Điểm ở mục 4 tính trên xác suất đã làm tròn 4 chữ số, đúng như file `dev_scores` được ghi, nên `ladder` và `collect` cho cùng số.

## 4. Kết quả

### Lượt 0: nhãn LLM chưa sửa (`dev_labels_sha = 7e5f9493db91`)

Chỉ để tham khảo. Chạy lại bằng `ladder` trên `dev.csv` của commit `cefae3a`.

| Mức | Cấu hình | Macro-F1 [KTC 95%] | F1 neg | F1 neu | F1 pos | Macro-F1 bình luận gốc | Macro-F1 trả lời |
|---|---|---|---|---|---|---|---|
| Mốc | `majority` | 0,213 [0,205; 0,222] | 0,640 | 0,000 | 0,000 | 0,212 | 0,214 |
| Mốc | `sopmi_oof` | 0,377 [0,348; 0,406] | 0,444 | 0,440 | 0,248 | 0,396 | 0,352 |
| Cổ điển | `tfidf_lr_text_oof` | 0,556 [0,524; 0,587] | 0,616 | 0,574 | 0,477 | 0,527 | 0,531 |
| Cổ điển | `tfidf_lr_ctx_oof` | 0,564 [0,532; 0,597] | 0,620 | 0,581 | 0,492 | 0,528 | 0,527 |
| Học sâu | `visobert_pre_text` | 0,531 [0,499; 0,561] | 0,677 | 0,399 | 0,517 | 0,649 | 0,442 |
| Học sâu | `visobert_pre_ctx` | 0,534 [0,502; 0,563] | 0,700 | 0,336 | 0,566 | 0,649 | 0,440 |

Ma trận nhầm lẫn (hàng: nhãn thật neg / neu / pos; cột: dự đoán):

- `visobert_pre_text`: [[372, 42, 104], [196, 120, 116], [13, 8, 129]]. Model dùng sẵn hiếm khi đoán `neu` (15,5% dự đoán, so với 39% nhãn). Phần lớn bình luận `neu` bị đoán là `neg` hoặc `pos`.
- `tfidf_lr_ctx_oof`: [[317, 149, 52], [147, 251, 34], [41, 32, 77]].

Nhận xét ban đầu: ViSoBERT dùng sẵn tốt hơn hẳn với bình luận gốc (0,649 so với 0,527 của TF-IDF), nhưng kém với bình luận trả lời (0,44). Có thể vì các bình luận trả lời thường là tranh luận, không có từ cảm xúc rõ.

### Sau 3 lượt rà soát (`dev_labels_sha = 7c078c6bb311`), các model chạy trên máy

| Mức | Cấu hình | Macro-F1 [KTC 95%] | F1 neg | F1 neu | F1 pos | Macro-F1 bình luận gốc | Macro-F1 trả lời |
|---|---|---|---|---|---|---|---|
| Mốc | `majority` | 0,215 [0,206; 0,224] | 0,645 | 0,000 | 0,000 | 0,214 | 0,216 |
| Mốc | `sopmi_oof` | 0,378 [0,349; 0,406] | 0,458 | 0,427 | 0,249 | 0,400 | 0,355 |
| Cổ điển | `tfidf_lr_text_oof` | 0,568 [0,534; 0,599] | 0,630 | 0,583 | 0,492 | 0,556 | 0,546 |
| Cổ điển | `tfidf_lr_ctx_oof` | 0,553 [0,520; 0,585] | 0,627 | 0,571 | 0,462 | 0,526 | 0,502 |
| Học sâu | `visobert_pre_text` | 0,530 [0,499; 0,562] | 0,679 | 0,399 | 0,514 | 0,647 | 0,443 |
| Học sâu | `visobert_pre_ctx` | 0,535 [0,505; 0,564] | 0,703 | 0,339 | 0,563 | 0,647 | 0,443 |

Ma trận nhầm lẫn (hàng: nhãn thật neg / neu / pos; cột `confusion` của `ladder_full.csv`):
- `tfidf_lr_text_oof`: [[323, 147, 54], [141, 250, 36], [37, 34, 78]]
- `visobert_pre_text`: [[375, 43, 106], [193, 119, 115], [13, 8, 128]]

Ablation `parent_text`: với TF-IDF, thêm ngữ cảnh lúc hơn một chút ở các lượt trước (0,564 so với 0,556), lúc kém ở nhãn cuối (0,553 so với 0,568). Mọi chênh lệch đều nằm trong khoảng tin cậy, nên chưa thấy ngữ cảnh giúp ích. Với ViSoBERT dùng sẵn, ngữ cảnh làm model đoán `neg` nhiều hơn (F1 `neg` tăng, F1 `neu` giảm), macro-F1 gần như không đổi.

### Đủ các mức, kể cả Kaggle (`dev_labels_sha = 7c078c6bb311`)

Chạy trên Kaggle (Tesla T4), 23/11/2026, `notebooks/sentiment_visobert.ipynb` và `notebooks/sentiment_qwen.ipynb` ở commit `2d41f4e`. Đặt các file kết quả vào `results/sentiment/kaggle/` rồi chạy `collect`: lệnh này chấm lại mọi file trên nhãn hiện tại. Dataset Kaggle đã dùng nhãn sau lượt 3: macro-F1 mà notebook Qwen tự ghi (`qwen7b_runs.json`, ví dụ `qwen7b_cot` 0,6358) khớp với điểm `collect` tính trên nhãn cuối.

| Mức | Cấu hình | Macro-F1 [KTC 95%] | F1 neg | F1 neu | F1 pos | Macro-F1 bình luận gốc | Macro-F1 trả lời |
|---|---|---|---|---|---|---|---|
| Học sâu | `visobert_ft_text_oof` | **0,717 [0,687; 0,747]** | 0,757 | 0,685 | 0,709 | 0,757 | 0,664 |
| Học sâu | `visobert_ft_ctx_oof` | 0,689 [0,658; 0,718] | 0,733 | 0,653 | 0,682 | 0,761 | 0,610 |
| LLM | `qwen7b_cot` | 0,636 [0,604; 0,665] | 0,744 | 0,533 | 0,630 | 0,698 | 0,566 |
| LLM | `qwen7b_few` | 0,623 [0,592; 0,652] | 0,726 | 0,530 | 0,611 | 0,685 | 0,550 |
| LLM | `qwen7b_few_ctx` | 0,604 [0,573; 0,634] | 0,718 | 0,519 | 0,574 | 0,703 | 0,512 |
| LLM | `qwen7b_zero` | 0,600 [0,571; 0,629] | 0,731 | 0,477 | 0,592 | 0,647 | 0,537 |
| Cổ điển | `tfidf_lr_text_oof` | 0,568 [0,534; 0,599] | 0,630 | 0,583 | 0,492 | 0,556 | 0,546 |

- ViSoBERT fine-tune, từng seed (42 / 43 / 44, trong `visobert_ft_runs.json`): `text` 0,708 / 0,718 / 0,718, tức **0,714 ± 0,006** (trung bình ± độ lệch chuẩn qua 3 seed); `ctx` 0,684 / 0,687 / 0,685 (0,686 ± 0,002). Dòng trong bảng là ensemble 3 seed (trung bình xác suất): `text` 0,717, `ctx` 0,689.
- Qwen chạy 1 lần (seed 42; suy luận dùng greedy nên không cần 3 seed). Thời gian trên T4 cho 1.100 mẫu (`qwen7b_runs.json`): zero 8,6 phút, few 21,4, few_ctx 29,1, cot 97,6.
- Bootstrap ghép cặp (1.000 lần, chênh macro-F1 [KTC 95%], tính bằng `compare` từ các file `dev_scores` của Kaggle): `visobert_ft_text_oof` hơn `visobert_ft_ctx_oof` 0,028 [0,005; 0,053], hơn `qwen7b_cot` 0,082 [0,049; 0,120], hơn `tfidf_lr_text_oof` 0,148 [0,111; 0,186].

Ma trận nhầm lẫn (hàng: nhãn thật neg / neu / pos) và tỷ lệ dự đoán neg / neu / pos (nhãn thật: 47,6% / 38,8% / 13,5%), cột `confusion` và `pred_share_*` của `ladder_full.csv`:
- `visobert_ft_text_oof`: [[390, 116, 18], [102, 297, 28], [15, 27, 107]]; 46,1% / 40,0% / 13,9%.
- `qwen7b_cot`: [[399, 78, 47], [146, 188, 93], [4, 12, 133]]; 49,9% / 25,3% / 24,8%.
- `qwen7b_few`: [[373, 112, 39], [123, 199, 105], [7, 13, 129]]; 45,7% / 29,5% / 24,8%.

Nhận xét:
- Fine-tune là bước nhảy lớn nhất: từ 0,530 (ViSoBERT dùng sẵn) lên 0,717, chủ yếu nhờ học được lớp `neu` (F1 từ 0,399 lên 0,685) theo cách `dev` định nghĩa.
- Qwen bắt `pos` tốt (recall 133/149 với CoT) nhưng gán `pos` và `neg` cho quá nhiều bình luận `neu`. Few-shot hơn zero-shot (+0,023), CoT hơn thêm một chút (+0,013) nhưng tốn gấp 4,5 lần thời gian. 18/1.100 lời giải thích CoT chuyển sang tiếng Trung giữa chừng (`collect` đếm từ `qwen7b_cot_outputs.jsonl`); nhãn không bị ảnh hưởng vì điểm lấy từ xác suất 3 chữ số sau "Nhãn:".
- Thêm `parent_text` không giúp ở mức nào: kém hơn với ViSoBERT fine-tune (chênh có ý nghĩa, chủ yếu ở bình luận trả lời: 0,664 xuống 0,610) và với Qwen few-shot (0,623 xuống 0,604), gần như không đổi với TF-IDF và ViSoBERT dùng sẵn. Có thể vì bình luận cha dài, chiếm phần lớn 256 token và lấn nội dung của chính bình luận.

## 5. Model cuối

**`visobert_ft_text`**: ViSoBERT fine-tune chỉ trên `text`, cùng siêu tham số như trên, huấn luyện lại trên toàn bộ 1.100 mẫu `dev` với 3 seed (42, 43, 44), lấy trung bình xác suất. Điểm ước lượng (ngoài fold): macro-F1 0,714 ± 0,006 cho từng seed, 0,717 [0,687; 0,747] cho ensemble 3 seed. Lý do chọn: tốt nhất và hơn mọi cấu hình khác một cách có ý nghĩa (bootstrap ghép cặp ở trên). Bản `ctx` phức tạp hơn mà lại kém hơn, nên không dùng.

`data/tasks/sentiment/corpus_labels.csv` (20.468 dòng, cùng thứ tự `id` với `corpus.csv`) có các cột `id, label, score`:
- `label` là argmax của model, trừ các mẫu thuộc `dev`, giữ nhãn `dev`.
- `score` là xác suất model gán cho `label` (với mẫu `dev` có nhãn khác dự đoán, là xác suất model gán cho nhãn `dev`), 4 chữ số thập phân.

Kết quả (`final` in ra):
- Tỷ lệ nhãn trên `corpus.csv`: `neg` 45,5%, `neu` 36,5%, `pos` 18,0%. Model học trên `dev`, vốn đã được chọn thêm mẫu dương (13,5% `pos`), nên tỷ lệ `pos` 18% trên corpus có thể bị đẩy lên.
- 1.103 dòng thuộc `dev` giữ nhãn `dev`. Model cuối khác nhãn đó ở 10 dòng (model đã học trên chính các mẫu này nên khớp gần hết, không dùng số này để đánh giá).
- Khớp với ViSoBERT dùng sẵn (`corpus_scores_visobert_pre_text.csv`) ở 69,8% bình luận (`label_agreement_with` do `final --from-file` in ra khi file `corpus_scores_visobert_pre_text.csv` của notebook nằm trong `results/sentiment/kaggle/`).
- Model cuối (safetensors) chỉ lưu trên máy của Dũng, không đưa vào git. File `corpus_labels.csv` hiện tại được chuyển từ định dạng cũ (có thêm `p_neg, p_neu, p_pos`) sang định dạng trên bằng chính các xác suất đó; nhãn không đổi.

```bash
# model chạy trên máy (chấm corpus rồi ghi corpus_labels.csv):
python -m src.classifiers.sentiment final <config>
# model chạy trên Kaggle: chạy phần cuối của notebook ViSoBERT, đặt các file corpus_scores_*.csv vào
# results/sentiment/kaggle/, rồi
python -m src.classifiers.sentiment final --from-file results/sentiment/kaggle/corpus_scores_visobert_ft_text.csv
```

## 6. Phân tích lỗi

Trên `visobert_ft_text_oof`: 306/1.100 mẫu sai. Hai chiều nhầm lẫn chiếm 71%: `neg` đoán thành `neu` 116 và `neu` đoán thành `neg` 102. Nhầm giữa `neg` và `pos` ít (33). Đọc các lỗi model tin chắc nhất (xác suất dự đoán từ 0,98 trở lên, lọc từ `dev_scores_visobert_ft_text_oof.csv`):

1. **Mỉa mai, câu hỏi vặn bị đoán `neu` hoặc `pos`.** Đây cũng là kiểu lỗi chính của nhãn ban đầu (mục 2). Đoán `neu`: 231, R141, R123, E083, 052 (câu hỏi vặn). Đoán `pos` vì bề mặt dùng từ tích cực để chê: 309, R277, 360.
2. **Bình luận nêu sự việc hoặc tranh luận bằng từ ngữ tiêu cực, nhãn `neu`, model đoán `neg`.** 026, E168, E222, E290, 089. Model dựa vào từ ngữ, không phân biệt được người viết đang giải thích hay đang bức xúc.
3. **Lời xin khoan hồng hay mong điều tốt trong bối cảnh buồn, nhãn `pos` theo quy ước 2, model đoán `neg`.** 081, 282. Bài viết về án tử hình kéo model về `neg`.
4. **Nhãn có thể còn sai hoặc mơ hồ.**
   - 211 (lượt 1 đã đổi `pos` thành `neu`; model đoán `pos` với xác suất 1,00).
   - 048 (lời khen chính quyền có thể là thật hoặc mỉa mai; nhãn `pos`, model đoán `neg`).
   - 153 (câu ngắn về chất lượng phim, nhãn `neu`, có thể là chê).
   - 476 (nhận xét ngắn về ngoại hình, nhãn `neg`, có thể là `neu`).

   Không sửa thêm, vì đã dừng rà soát sau lượt 3 và theo quy ước 4. Nếu có người gán độc lập, nên xem lại các mẫu này trước.

Điểm mạnh, điểm yếu của từng mức:

| Mức | Macro-F1 | Cần nhãn | Chi phí | Ghi chú |
|---|---|---|---|---|
| SO-PMI | 0,378 | không (chỉ 14 từ hạt giống mỗi cực, 2 ngưỡng chọn trên `dev`) | CPU (trong `ladder`) | Từ điển học từ corpus không phân biệt được `neu`, gần như bỏ sót `pos` (F1 0,249). |
| TF-IDF + LR | 0,568 | có (khoảng 880 mẫu mỗi fold) | CPU (trong `ladder`) | Ổn định, dễ giải thích, nhưng yếu nhất ở `pos` (ít mẫu). |
| ViSoBERT dùng sẵn | 0,530 | không | CPU (trong `ladder`) | Tốt với bình luận gốc, hiếm khi đoán `neu`: định nghĩa `neu` của model khác của `dev`. |
| Qwen2.5-7B | 0,600 đến 0,636 | không (few-shot chỉ 9 ví dụ) | 9 đến 98 phút trên T4 cho 1.100 mẫu, nên gán cả corpus sẽ mất nhiều giờ | Đoán `pos` quá tay; không cần huấn luyện; sửa định nghĩa trong prompt là đổi được hành vi. |
| ViSoBERT fine-tune | **0,717** | có (khoảng 880 mẫu mỗi fold) | cần GPU | Tốt nhất ở mọi lớp. Vẫn sai với mỉa mai, và học luôn cả các nhãn còn sai trong `dev`. |

Lưu ý chung: mọi điểm số đo mức khớp với nhãn LLM đã được LLM rà lại (xem đầu file). Model fine-tune học trực tiếp các nhãn này, nên có lợi thế khi chấm bằng chính chúng so với Qwen hay ViSoBERT dùng sẵn.

## 7. Chạy lại

```bash
pip install -r requirements.txt   # Python 3.9
python -m src.classifiers.sentiment ladder
# Kaggle: tạo dataset private từ data/tasks/sentiment/, chạy notebooks/sentiment_visobert.ipynb và
# notebooks/sentiment_qwen.ipynb (GPU, bật Internet), tải file kết quả về results/sentiment/kaggle/
python -m src.classifiers.sentiment collect
python -m src.classifiers.sentiment compare visobert_ft_text_oof visobert_ft_ctx_oof   # tương tự với qwen7b_cot, tfidf_lr_text_oof
# phần cuối notebook ViSoBERT ghi corpus_scores_visobert_ft_text.csv và corpus_scores_visobert_pre_text.csv, rồi
python -m src.classifiers.sentiment final --from-file results/sentiment/kaggle/corpus_scores_visobert_ft_text.csv
```

## 8. Độ đồng thuận với người

Chưa làm. Theo `TASK.md` (mục "Đo độ đồng thuận với người"):
1. Chạy `python -m src.classifiers.agreement sample sentiment` để lấy 100 mẫu `dev` ngẫu nhiên, phân tầng theo nhãn, seed 42, ghi vào `data/tasks/sentiment/agreement.csv` với cột `label` để trống.
2. Một thành viên khác (đề xuất: Tấn) gán cột `label` bằng tay theo định nghĩa nhãn, không xem `dev.csv` và không dùng công cụ AI.
3. Chạy `python -m src.classifiers.agreement score sentiment` để tính tỷ lệ khớp và Cohen's kappa, rồi ghi kết quả vào mục này: số mẫu, tỷ lệ khớp, kappa, các `id` lệch và nhận xét. Không sửa `dev.csv` theo kết quả này. Nếu kappa dưới khoảng 0,6, báo cả nhóm trước khi dùng nhãn cho báo cáo.
