# Cảm xúc (sentiment): ghi chép

Người phụ trách: Dũng. Branch: `sentiment-dung`. Code: `src/classifiers/sentiment.py`, `notebooks/sentiment_visobert.ipynb`, `notebooks/sentiment_qwen.ipynb`.

> Mỗi bảng điểm ghi kèm `dev_labels_sha` (12 ký tự đầu của SHA-256 trên cặp `id,label` của `dev.csv`, do `ladder` và `collect` in ra), để biết điểm tính trên nhãn của lượt nào.

## 1. Dữ liệu

- `dev.csv`: 1.100 bình luận VnExpress. Nhãn ban đầu do LLM gán (`dev_labels_sha = 7e5f9493db91`): 518 `neg`, 432 `neu`, 150 `pos`.
  - 63% là bình luận trả lời (có `parent_text`), so với 36,6% trong `corpus.csv`. Tập `dev` cũng đã được chọn thêm mẫu dương, nên không dùng để ước lượng tỷ lệ nhãn.
  - `id` của `dev` là mã riêng (`001`, `E288`, ...), không phải `comment_id`. Để nối với `corpus.csv`, ghép theo (`text`, `parent_text`): cả 1.100 mẫu đều khớp, 2 mẫu khớp với hơn một bình luận giống hệt nhau (tổng 1.103 dòng corpus).
- `corpus.csv`: 20.468 bình luận.
- Không có `train.csv`: model học từ `dev` được chấm ngoài fold (5 fold, `StratifiedKFold(shuffle=True, random_state=42)`, siêu tham số chọn bằng kiểm định chéo lồng trong phần huấn luyện).
- Phân bố nhãn sau khi sửa: _(điền khi xong)_

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

| Lượt | Model chọn mẫu | Kiểm tra (chắc neg / neu / pos / phân vân) | Sửa | Chiều sửa | Commit |
|---|---|---|---|---|---|
| 1 | `visobert_pre_text` (không học từ `dev`) | 10 / 10 / 10 / 10 | | | |

Ví dụ và kiểu lỗi hay gặp: _(điền theo từng lượt, ghi `id`)_

Trường hợp định nghĩa còn mơ hồ: _(ghi `id` và cách đã quyết)_

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

### Sau khi nhãn ổn định

_(bảng của `collect`, ghi `dev_labels_sha`)_

## 5. Model cuối

_(model, lý do, điểm cuối, tỷ lệ nhãn dự đoán trên `corpus.csv`)_

```bash
# model chạy trên máy:
python -m src.classifiers.sentiment predict-corpus <config>
python -m src.classifiers.sentiment label-corpus results/sentiment/corpus_scores_<config>.csv
# model chạy trên Kaggle: chạy phần cuối của notebook, rồi
python -m src.classifiers.sentiment label-corpus results/sentiment/kaggle/corpus_scores_<config>.csv
```

`corpus_labels.csv` có các cột `id, label, score, p_neg, p_neu, p_pos`. `label` là argmax của model, trừ các mẫu thuộc `dev` giữ nhãn đã sửa tay. `score = p_pos − p_neg` (từ −1 đến 1) luôn là điểm của model.

## 6. Phân tích lỗi

_(kiểu lỗi kèm `id`; điểm mạnh, điểm yếu của từng mức về độ chính xác, chi phí, lượng nhãn cần)_

## 7. Chạy lại

```bash
pip install -r requirements.txt   # Python 3.9
python -m src.classifiers.sentiment ladder
# Kaggle: tạo dataset private từ data/tasks/sentiment/, chạy notebooks/sentiment_visobert.ipynb và
# notebooks/sentiment_qwen.ipynb (GPU, bật Internet), tải file kết quả về results/sentiment/kaggle/
python -m src.classifiers.sentiment collect
```
