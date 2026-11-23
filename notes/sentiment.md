# Cảm xúc (sentiment): ghi chép

> **Nhãn `dev` của bài này là nhãn LLM, đã được LLM rà lại, không phải nhãn người kiểm tra.** Nhãn ban đầu do Claude gán. Ba lượt rà soát (120 mẫu) cũng do Claude (`claude-opus-5-5`) quyết, theo ngoại lệ ghi ở `TASK.md` mục 4, do Dũng quyết định ngày 22/11/2026. Vì vậy mọi điểm số dưới đây đo mức khớp với nhãn LLM, không phải với người đọc thật. Muốn ước lượng chất lượng nhãn, cần một người gán độc lập một mẫu ngẫu nhiên rồi tính độ đồng thuận.

Người phụ trách: Dũng. Branch: `sentiment-dung`. Code: `src/classifiers/sentiment.py`, `notebooks/sentiment_visobert.ipynb`, `notebooks/sentiment_qwen.ipynb`.

> Mỗi bảng điểm ghi kèm `dev_labels_sha` (12 ký tự đầu của SHA-256 trên cặp `id,label` của `dev.csv`, do `ladder` và `collect` in ra), để biết điểm tính trên nhãn của lượt nào.

## 1. Dữ liệu

- `dev.csv`: 1.100 bình luận VnExpress. Nhãn ban đầu do LLM gán (`dev_labels_sha = 7e5f9493db91`): 518 `neg`, 432 `neu`, 150 `pos`.
  - 63% là bình luận trả lời (có `parent_text`), so với 36,6% trong `corpus.csv`. Tập `dev` cũng đã được chọn thêm mẫu dương, nên không dùng để ước lượng tỷ lệ nhãn.
  - `id` của `dev` là mã riêng (`001`, `E288`, ...), không phải `comment_id`. Để nối với `corpus.csv`, ghép theo (`text`, `parent_text`): cả 1.100 mẫu đều khớp, 2 mẫu khớp với hơn một bình luận giống hệt nhau (tổng 1.103 dòng corpus).
- `corpus.csv`: 20.468 bình luận.
- Không có `train.csv`: model học từ `dev` được chấm ngoài fold (5 fold, `StratifiedKFold(shuffle=True, random_state=42)`, siêu tham số chọn bằng kiểm định chéo lồng trong phần huấn luyện).
- Phân bố nhãn sau 3 lượt rà soát (`dev_labels_sha = 7c078c6bb311`): 524 `neg`, 427 `neu`, 149 `pos`. Tổng cộng 7/1.100 nhãn đổi so với ban đầu (`git diff cefae3a -- data/tasks/sentiment/dev.csv`).

## 2. Các lượt sửa nhãn

Quy trình mỗi lượt:

```bash
python -m src.classifiers.sentiment review --scores <dev_scores của model tốt nhất hiện có>
python -m src.classifiers.sentiment label N      # gán trong terminal: 1 neg / 2 neu / 3 pos, tự lưu, chạy lại để làm tiếp
# (hoặc mở notes/sentiment_review/round_N.csv và tự điền cột my_label theo TASK.md mục 6)
# Cả hai cách đều không hiện nhãn hiện tại và dự đoán của model, để tự quyết trước.
python -m src.classifiers.sentiment apply-review N           # xem các mẫu khác nhãn hiện tại
python -m src.classifiers.sentiment apply-review N --write   # ghi vào dev.csv (chỉ đổi cột label)
git add data/tasks/sentiment/dev.csv notes/sentiment_review/ && git commit -m "dev: luot N, kiem tra 40 mau, sua K nhan"
python -m src.classifiers.sentiment ladder && python -m src.classifiers.sentiment collect
```

`round_N_key.csv` ghi nhóm của từng mẫu (tin chắc `neg`/`neu`/`pos`, phân vân), dự đoán của model và nhãn trước khi sửa. `round_N_summary.json` ghi số mẫu đã kiểm tra, số nhãn sửa theo nhóm và theo chiều.

Người gán ở cả ba lượt: Claude (`labeled_by = claude-opus-5-5`). Lý do của từng mẫu nằm ở cột `reason` của `round_N.csv`.

| Lượt | Model chọn mẫu | Kiểm tra (chắc neg / neu / pos / phân vân) | Sửa (giữ lại sau lượt 3) | Chiều sửa (giữ lại) | Model khớp người gán | Commit |
|---|---|---|---|---|---|---|
| 1 | `visobert_pre_text` (không học từ `dev`) | 10 / 10 / 10 / 10 | 3 (2) | neu→neg (R123), pos→neu (211) | 28/40 | `ec7c0dd` |
| 2 | `tfidf_lr_ctx_oof` (tốt nhất sau lượt 1) | 10 / 10 / 10 / 10 | 5 (3) | neu→neg (R212, 267, 180) | 25/40 | `eb21099` |
| 3 | `tfidf_lr_ctx_oof` (tốt nhất sau lượt 2) | 10 / 10 / 10 / 10 | 2 (2) | neu→neg (001, 147) | 24/40 | `fbdbece` |

Ở lượt 3 có thêm quy ước "mẫu còn mơ hồ thì giữ nhãn hiện tại", nên 3 thay đổi của lượt 1-2 được trả về nhãn cũ: 202 (neg), E089 (pos), R254 (neg). `round_1_summary.json` và `round_2_summary.json` ghi số liệu lúc chạy lượt đó, trước khi trả về. Cột "giữ lại" là số cuối cùng.

**Dừng sau lượt 3:** lượt này chỉ sửa 2/40 mẫu, đúng điều kiện dừng ở `TASK.md` (khoảng 1-2 trên 30-40). Điểm của model tốt nhất cũng không đổi đáng kể qua các lượt (0,564 → 0,568 → 0,570 → 0,568), đều nằm trong khoảng tin cậy.

**Kiểu lỗi hay gặp của nhãn ban đầu:** gần như mọi chỗ sửa đều theo chiều `neu → neg`.
- Câu hỏi vặn hoặc mỉa mai, bề ngoài là câu hỏi hay câu ngắn nên bị gán `neu`: R123 "Bạn có bán hàng trên sàn chưa ?", 180 "Vậy theo bạn giờ đổ cho ai?", 267 "Giờ chả 10 làn".
- Lời lo ngại nói nhẹ: 001 "Mong là sẽ không thay đổi quá nhanh, theo ko kịp", 147 (khen cà phê rồi lo giá rớt).
- Lập luận rằng một việc là bất công: R212.

**Quy ước thêm khi định nghĩa ở mục 6 chưa rõ** (áp dụng cho các lượt sau và khi rà lại):
1. Câu hỏi vặn, lời coi thường ("có đọc bài không", "bạn hiểu chưa", "bạn làm đạo diễn đi"): `neg`. Phản biện có lý lẽ, đính chính số liệu mà không hạ thấp người kia: `neu` (ví dụ 307, E299, 132).
2. Lời mong, chúc điều tốt không kèm than phiền: `pos` (E288, 044, 154). Lời mong kèm buồn, cảm thông hay lo lắng: `neg` (008, 093, 102). Lời mong mang tính yêu cầu: `neu` (436).
3. Có ý khen trước rồi ý chính sau "thế nhưng" là lo, chê: `neg` (147, 275).
4. Mẫu còn mơ hồ sau khi đọc kỹ thì giữ nhãn hiện tại. Ví dụ E240 "Là an toàn tuyệt đối nha bạn" (trấn an hay mỉa mai?), 202, E089.

**Rà lại cùng kiểu (sau lượt 3):** tìm trong các mẫu `dev` chưa kiểm tra. 9 mẫu coi thường kiểu "đọc bài đi" đều đã là `neg`. 8 mẫu mở đầu bằng "mong / hy vọng / chúc" đã khớp quy ước 2. Không cần sửa thêm.

## 3. Các model

| Mức | Cấu hình | Đầu vào | Học từ | Siêu tham số | Chạy ở đâu, thời gian |
|---|---|---|---|---|---|
| Mốc | `majority` | — | đếm lớp trên `dev` | — | máy cá nhân |
| Mốc | `sopmi_oof` | `text` | từ điển SO-PMI trên `corpus.csv` (không nhãn); 2 ngưỡng chọn trên 4/5 `dev` | 14 từ hạt giống mỗi cực, âm tiết + bigram âm tiết, df ≥ 5 | máy cá nhân |
| Cổ điển | `tfidf_lr_text_oof` | `text` | `dev` (ngoài fold) | TF-IDF từ 1-2 gram + ký tự 2-5 gram; LR, `C` ∈ {0,3; 1; 3; 10}, `class_weight` ∈ {None, balanced}, chọn bằng CV lồng | máy cá nhân |
| Cổ điển | `tfidf_lr_ctx_oof` | `text` + `parent_text` | `dev` (ngoài fold) | như trên, thêm khối TF-IDF riêng cho `parent_text` | máy cá nhân |
| Học sâu | `visobert_pre_text` | `text` | không huấn luyện | `5CD-AI/Vietnamese-Sentiment-visobert`; đầu ra theo thứ tự `NEG, POS, NEU` (`config.id2label`), đã đổi về `neg, neu, pos` | máy cá nhân, CPU |
| Học sâu | `visobert_pre_ctx` | cặp (`text`, `parent_text`) | không huấn luyện | như trên | máy cá nhân, CPU |
| Học sâu | `visobert_ft_{text,ctx}_oof` | như trên | `dev` (ngoài fold) | 4 epoch, lr 2e-5, batch 16, max_len 256, 3 seed (42, 43, 44); cố định trước, không chỉnh trên `dev` | Kaggle |
| LLM | `qwen7b_{zero,few,cot}` | `text` | không huấn luyện; few-shot lấy 3 ví dụ mỗi lớp từ 4 fold còn lại | Qwen2.5-7B-Instruct 4-bit nf4, nhãn là chữ số 0/1/2, điểm = softmax trên 3 token | Kaggle |
| LLM | `qwen7b_few_ctx` | `text` + `parent_text` | như trên | như trên | Kaggle |

Máy cá nhân: macOS, 11 lõi CPU, Python 3.9 (`/usr/bin/python3`), torch 2.8. `ladder` chạy hết khoảng 1,5 phút.

## 4. Kết quả

### Lượt 0: nhãn LLM chưa sửa (`dev_labels_sha = 7e5f9493db91`)

Chỉ để tham khảo. Các con số này sẽ thay đổi khi nhãn được sửa.

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
| Cổ điển | `tfidf_lr_ctx_oof` | 0,553 [0,519; 0,585] | 0,626 | 0,571 | 0,461 | 0,524 | 0,502 |
| Học sâu | `visobert_pre_text` | 0,530 [0,499; 0,562] | 0,679 | 0,399 | 0,514 | 0,647 | 0,443 |
| Học sâu | `visobert_pre_ctx` | 0,535 [0,505; 0,564] | 0,703 | 0,339 | 0,563 | 0,647 | 0,443 |

Ma trận nhầm lẫn (hàng: nhãn thật neg / neu / pos):
- `tfidf_lr_text_oof`: [[323, 147, 54], [141, 250, 36], [37, 34, 78]]
- `visobert_pre_text`: [[375, 43, 106], [193, 119, 115], [13, 8, 128]]

Ablation `parent_text`: với TF-IDF, thêm ngữ cảnh lúc hơn một chút ở các lượt trước (0,564 so với 0,556), lúc kém ở nhãn cuối (0,553 so với 0,568). Mọi chênh lệch đều nằm trong khoảng tin cậy, nên chưa thấy ngữ cảnh giúp ích. Với ViSoBERT dùng sẵn, ngữ cảnh làm model đoán `neg` nhiều hơn (F1 `neg` tăng, F1 `neu` giảm), macro-F1 gần như không đổi.

### Đủ các mức, kể cả Kaggle (`dev_labels_sha = 7c078c6bb311`)

Chạy trên Kaggle (Tesla T4), 23/11/2026, `notebooks/sentiment_visobert.ipynb` và `notebooks/sentiment_qwen.ipynb` ở commit `2d41f4e`. `collect` chấm lại mọi file trên nhãn hiện tại. Dataset Kaggle đã dùng nhãn sau lượt 3: macro-F1 mà notebook Qwen tự ghi (`qwen7b_runs.json`, ví dụ `qwen7b_cot` 0,6358) khớp với điểm tính trên nhãn cuối, không khớp với nhãn ban đầu (0,6364).

| Mức | Cấu hình | Macro-F1 [KTC 95%] | F1 neg | F1 neu | F1 pos | Macro-F1 bình luận gốc | Macro-F1 trả lời |
|---|---|---|---|---|---|---|---|
| Học sâu | `visobert_ft_text_oof` | **0,717 [0,687; 0,747]** | 0,757 | 0,685 | 0,709 | 0,757 | 0,664 |
| Học sâu | `visobert_ft_ctx_oof` | 0,689 [0,658; 0,718] | 0,733 | 0,653 | 0,682 | 0,761 | 0,610 |
| LLM | `qwen7b_cot` | 0,636 [0,604; 0,665] | 0,744 | 0,533 | 0,630 | 0,698 | 0,566 |
| LLM | `qwen7b_few` | 0,623 [0,592; 0,652] | 0,726 | 0,530 | 0,611 | 0,685 | 0,550 |
| LLM | `qwen7b_few_ctx` | 0,604 [0,573; 0,634] | 0,718 | 0,519 | 0,574 | 0,703 | 0,512 |
| LLM | `qwen7b_zero` | 0,600 [0,571; 0,629] | 0,731 | 0,477 | 0,592 | 0,647 | 0,537 |
| Cổ điển | `tfidf_lr_text_oof` | 0,568 [0,534; 0,599] | 0,630 | 0,583 | 0,492 | 0,556 | 0,546 |

- ViSoBERT fine-tune, từng seed (42 / 43 / 44): `text` 0,708 / 0,718 / 0,718 (0,714 ± 0,006), `ctx` 0,684 / 0,687 / 0,685 (0,686 ± 0,002). Dòng trong bảng là trung bình xác suất của 3 seed.
- Qwen chạy 1 lần (seed 42; suy luận dùng greedy nên không cần 3 seed). Thời gian trên T4 cho 1.100 mẫu: zero 8,6 phút, few 21,4, few_ctx 29,1, cot 97,6.
- Bootstrap ghép cặp (1.000 lần, chênh macro-F1 [KTC 95%]): `visobert_ft_text_oof` hơn `visobert_ft_ctx_oof` 0,028 [0,005; 0,053], hơn `qwen7b_cot` 0,082 [0,049; 0,120], hơn `tfidf_lr_text_oof` 0,148 [0,111; 0,186].

Ma trận nhầm lẫn (hàng: nhãn thật neg / neu / pos) và tỷ lệ dự đoán neg / neu / pos (nhãn thật: 47,6% / 38,8% / 13,5%):
- `visobert_ft_text_oof`: [[390, 116, 18], [102, 297, 28], [15, 27, 107]]; 46,1% / 40,0% / 13,9%.
- `qwen7b_cot`: [[399, 78, 47], [146, 188, 93], [4, 12, 133]]; 49,9% / 25,3% / 24,8%.
- `qwen7b_few`: [[373, 112, 39], [123, 199, 105], [7, 13, 129]]; 45,7% / 29,5% / 24,8%.

Nhận xét:
- Fine-tune là bước nhảy lớn nhất: từ 0,530 (ViSoBERT dùng sẵn) lên 0,717, chủ yếu nhờ học được lớp `neu` (F1 0,399 → 0,685) theo cách `dev` định nghĩa.
- Qwen bắt `pos` tốt (recall 133/149 với CoT) nhưng gán `pos` và `neg` cho quá nhiều bình luận `neu`. Few-shot hơn zero-shot (+0,023), CoT hơn thêm một chút (+0,013) nhưng tốn gấp 4,5 lần thời gian. 18/1.100 lời giải thích CoT chuyển sang tiếng Trung giữa chừng; nhãn không bị ảnh hưởng vì điểm lấy từ xác suất 3 chữ số sau "Nhãn:".
- Thêm `parent_text` không giúp ở mức nào: kém hơn với ViSoBERT fine-tune (chênh có ý nghĩa, chủ yếu ở bình luận trả lời: 0,664 → 0,610) và với Qwen few-shot (0,623 → 0,604), gần như không đổi với TF-IDF và ViSoBERT dùng sẵn. Có thể vì bình luận cha dài, chiếm phần lớn 256 token và lấn nội dung của chính bình luận.

## 5. Model cuối

**`visobert_ft_text`**: ViSoBERT fine-tune chỉ trên `text`, cùng siêu tham số như trên, huấn luyện lại trên toàn bộ 1.100 mẫu `dev` với 3 seed (42, 43, 44), lấy trung bình xác suất. Điểm ước lượng: macro-F1 ngoài fold 0,717 [0,687; 0,747]. Lý do chọn: tốt nhất và hơn mọi cấu hình khác một cách có ý nghĩa (bootstrap ghép cặp ở trên). Bản `ctx` phức tạp hơn mà lại kém hơn, nên không dùng.

Đã ghi `data/tasks/sentiment/corpus_labels.csv` (20.468 dòng, đủ mọi `id` của `corpus.csv`):
- Tỷ lệ nhãn trên `corpus.csv`: `neg` 45,5%, `neu` 36,5%, `pos` 18,0%.
- 1.103 dòng thuộc `dev` giữ nhãn đã rà soát. Model cuối khác nhãn đó ở 10 dòng (model đã học trên chính các mẫu này nên khớp gần hết, không dùng số này để đánh giá).
- Khớp với ViSoBERT dùng sẵn (`corpus_scores_visobert_pre_text.csv`) ở 69,8% bình luận.
- Model lưu trên máy ở `results/results-visobert/visobert_sentiment_text/` (safetensors, không đưa vào git).

```bash
# model chạy trên máy:
python -m src.classifiers.sentiment predict-corpus <config>
python -m src.classifiers.sentiment label-corpus results/sentiment/corpus_scores_<config>.csv
# model chạy trên Kaggle: chạy phần cuối của notebook, rồi
python -m src.classifiers.sentiment label-corpus results/sentiment/kaggle/corpus_scores_<config>.csv
```

`corpus_labels.csv` có các cột `id, label, score, p_neg, p_neu, p_pos`. `label` là argmax của model, trừ các mẫu thuộc `dev` giữ nhãn đã sửa tay. `score = p_pos − p_neg` (từ −1 đến 1) luôn là điểm của model.

## 6. Phân tích lỗi

Trên `visobert_ft_text_oof`: 306/1.100 mẫu sai. Hai chiều nhầm lẫn chiếm 71%: `neg → neu` 116 và `neu → neg` 102. Nhầm giữa `neg` và `pos` ít (33). Đọc 30 lỗi model tin chắc nhất (xác suất ≥ 0,98):

1. **Mỉa mai, câu hỏi vặn bị đoán `neu` hoặc `pos`.** Đây cũng là kiểu lỗi chính của nhãn ban đầu (mục 2). `neu`: 231 "đạo diễn có vai trò gì trong phim này nhỉ???", R141, R123, E083 "Xe điện phát miễn phí hả bạn?", 052. `pos` vì bề mặt dùng từ tích cực: 309 "Một bước lùi đột phá của điện ảnh Việt", R277 "Đẹp nhất thì quá rồi", 360 "Đang chờ bạn vươn tầm quốc tế đó!".
2. **Bình luận nêu sự việc hoặc tranh luận bằng từ ngữ tiêu cực, nhãn `neu`, model đoán `neg`.** 026 "Thiếu trường học, thiếu cơ sở vật chất...", E168 (giải thích vì sao làm đường chậm), E222, E290, 089 "Từ lỗi hành chính chuyển sang hình sự rồi.". Model dựa vào từ ngữ, không phân biệt được người viết đang giải thích hay đang bức xúc.
3. **Lời xin khoan hồng hay mong điều tốt trong bối cảnh buồn, nhãn `pos` theo quy ước 2, model đoán `neg`.** 081 "Trời không cho bà ấy chết thì hãy để bà ấy sống...", 282. Bài viết về án tử hình kéo model về `neg`.
4. **Nhãn có thể còn sai hoặc mơ hồ.**
   - 211 (lượt 1 đã đổi `pos → neu`; model đoán `pos` với xác suất 1,00).
   - 048 "Chính quyền xã La Dạ phản ứng nhanh quá" (`pos`, model đoán `neg`).
   - 153 "Phim thế còn kêu nhạt" (`neu`).
   - 476 "Nhan sắc bình thường." (`neg`).

   Không sửa thêm, vì đã dừng rà soát sau lượt 3 và theo quy ước 4. Nếu có người gán độc lập, nên xem lại các mẫu này trước.

Điểm mạnh, điểm yếu của từng mức:

| Mức | Macro-F1 | Cần nhãn | Chi phí | Ghi chú |
|---|---|---|---|---|
| SO-PMI | 0,378 | không (chỉ 14 từ hạt giống mỗi cực, 2 ngưỡng chọn trên `dev`) | CPU (cả `ladder` khoảng 1,5 phút) | Từ điển học từ corpus không phân biệt được `neu`, gần như bỏ sót `pos` (F1 0,249). |
| TF-IDF + LR | 0,568 | có (~880 mẫu mỗi fold) | CPU (trong `ladder`) | Ổn định, dễ giải thích, nhưng yếu nhất ở `pos` (ít mẫu). |
| ViSoBERT dùng sẵn | 0,530 | không | CPU (trong `ladder`) | Tốt với bình luận gốc, hiếm khi đoán `neu`: định nghĩa `neu` của model khác của `dev`. |
| Qwen2.5-7B | 0,600–0,636 | không (few-shot chỉ 9 ví dụ) | 9–98 phút trên T4 cho 1.100 mẫu, nên gán cả corpus sẽ mất nhiều giờ | Đoán `pos` quá tay; không cần huấn luyện; sửa định nghĩa trong prompt là đổi được hành vi. |
| ViSoBERT fine-tune | **0,717** | có (~880 mẫu mỗi fold) | cần GPU | Tốt nhất ở mọi lớp. Vẫn sai với mỉa mai, và học luôn cả các nhãn còn sai trong `dev`. |

Lưu ý chung: mọi điểm số đo mức khớp với nhãn LLM đã được LLM rà lại (xem đầu file). Model fine-tune học trực tiếp các nhãn này, nên có lợi thế khi chấm bằng chính chúng so với Qwen hay ViSoBERT dùng sẵn.

## 7. Chạy lại

```bash
pip install -r requirements.txt   # Python 3.9
python -m src.classifiers.sentiment ladder
# Kaggle: tạo dataset private từ data/tasks/sentiment/, chạy notebooks/sentiment_visobert.ipynb và
# notebooks/sentiment_qwen.ipynb (GPU, bật Internet), tải file kết quả về results/sentiment/kaggle/
python -m src.classifiers.sentiment collect
# phần cuối notebook ViSoBERT ghi corpus_scores_visobert_ft_text.csv, rồi
python -m src.classifiers.sentiment label-corpus results/sentiment/kaggle/corpus_scores_visobert_ft_text.csv
```
