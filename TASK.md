# Phân công: 3 bài toán phân loại

Mỗi thành viên phụ trách một bài toán. Mục tiêu chung:
- Có một bộ nhãn `dev` sạch.
- Có một model tốt nhất, chọn ra sau khi so sánh nhiều mức model.
- Có nhãn do model đó gán cho toàn bộ dữ liệu.
- Có một file ghi chép đầy đủ quá trình.

| Bài toán | Gán gì | Dữ liệu | Người phụ trách |
|---|---|---|---|
| Clickbait | Tiêu đề bài báo: `1` clickbait, `0` không | `data/tasks/clickbait/` | Kiên |
| Cảm xúc | Bình luận: `neg` tiêu cực, `neu` trung tính, `pos` tích cực | `data/tasks/sentiment/` | Dũng |
| Thiếu văn minh | Bình luận: `1` thiếu văn minh, `0` văn minh | `data/tasks/incivility/` | Tấn |

**Cần nộp khi xong** (một pull request vào `curr`):
- [ ] `data/tasks/<bai-toan>/dev.csv` với nhãn đã sửa. Mỗi lượt sửa là một commit riêng.
- [ ] Code chạy được: `src/classifiers/<bai-toan>.py` cho các model chạy trên máy, `notebooks/<bai-toan>_<model>.ipynb` cho các model chạy trên Kaggle.
- [ ] `data/tasks/<bai-toan>/corpus_labels.csv`: nhãn của model cuối cho toàn bộ `corpus.csv`.
- [ ] `notes/<bai-toan>.md`: file ghi chép theo mục 7.
- [ ] `data/tasks/<bai-toan>/agreement.csv`: 100 mẫu `dev` do một người gán độc lập (mục 8).

**Thứ tự làm việc:**
1. Chạy các model rẻ để có điểm ban đầu.
2. Lặp sửa nhãn `dev` (mục 4) cho tới khi ổn định.
3. Chạy đủ các mức model và ablation trên nhãn đã ổn định (mục 5).
4. Chọn model cuối và gán nhãn cho `corpus.csv`.
5. Viết file ghi chép và mở pull request.
6. Đo độ đồng thuận giữa nhãn `dev` và một người gán độc lập (mục 8), ghi vào file ghi chép.

## 1. Bắt đầu

```bash
git checkout curr
git pull
git checkout -b <bai-toan>-<ten>        # ví dụ: clickbait-kien, sentiment-dung, incivility-tan
pip install -r requirements.txt         # Python 3.9
```

Mọi việc làm trên branch riêng. Xong thì mở pull request vào `curr`.

## 2. Dữ liệu của mỗi bài toán

| File | Nội dung | Cột |
|---|---|---|
| `dev.csv` | Mẫu VnExpress đã có nhãn. Nhãn ban đầu do một LLM (Claude) gán theo định nghĩa ở mục 6, **chưa được người kiểm tra** | `id`, nội dung, `label` |
| `train.csv` | Dữ liệu công khai để huấn luyện (bài toán cảm xúc không có) | `id`, nội dung, `label` |
| `corpus.csv` | Toàn bộ dữ liệu cần gán nhãn ở bước cuối | `id`, nội dung |

- **Nội dung:**
  - Clickbait: `title`, `sapo`.
  - Bình luận: `text`, `parent_text`. `parent_text` là bình luận mà bình luận này trả lời; để trống nếu là bình luận gốc.
- **Trong bình luận:**
  - `@USER_xxxxxxxxxxxx` là người được nhắc tên, đã ẩn danh.
  - `[PHONE]` và `[EMAIL]` là thông tin đã che.
- **Số lượng:**

| Bài toán | `dev.csv` | `train.csv` | `corpus.csv` |
|---|---|---|---|
| Clickbait | 750 tiêu đề (96 clickbait) | 3.414 tiêu đề ViClickbait-2025 (31% clickbait, 8 báo, cột `source` ghi tên báo) | 1.244 bài |
| Cảm xúc | 1.100 bình luận (518 neg, 432 neu, 150 pos) | Không có | 20.468 bình luận |
| Thiếu văn minh | 1.100 bình luận (170 thiếu văn minh) | 10.000 bình luận UIT-ViCTSD (nhãn toxic, khoảng 11%), coi toxic là thiếu văn minh | 20.468 bình luận |

**Lưu ý về `dev.csv`:**
- Tập này không đại diện cho toàn bộ dữ liệu, vì có một phần mẫu được chọn thêm để tăng số mẫu dương. Đừng dùng nó để ước lượng tỷ lệ nhãn.
- Không có tập test riêng. Báo cáo điểm như sau:
  - Model **không** học từ `dev`: chấm thẳng trên toàn bộ `dev`.
  - Model **có** học từ `dev`: dùng kiểm định chéo 5 phần (`StratifiedKFold`, `random_state=42`). Mỗi mẫu chỉ được chấm bởi model không thấy nó khi huấn luyện. Siêu tham số chọn bằng kiểm định chéo lồng bên trong phần huấn luyện.

Hàm đọc dữ liệu có sẵn trong `src/classifiers/common.py`: `load_dev(task)`, `load_train(task)`, `load_corpus(task)`. Ở đó cũng có sẵn hàm tính độ đo và khoảng tin cậy bootstrap.

## 3. Dữ liệu và model: đã có gì, cần tải gì

**Dữ liệu: đã tải và đã chuẩn bị xong**, nằm sẵn trong repo, không cần tải lại.

| Bộ dữ liệu | Nguồn | Đã thành |
|---|---|---|
| ViClickbait-2025 | Mendeley Data, DOI 10.17632/3wc46bfcjc.1, CC BY 4.0. Chỉ lấy file CSV, bỏ 3.410 ảnh | `data/external/viclickbait2025/`, rồi `data/tasks/clickbait/train.csv` |
| UIT-ViCTSD | Hugging Face `tarudesu/ViCTSD`, do tác giả đầu của bài báo đăng, chỉ dùng cho nghiên cứu. Gộp cả 3 file train, valid, test | `data/external/victsd/`, rồi `data/tasks/incivility/train.csv` |
| Bình luận VnExpress | Nhóm tự thu thập, đã ẩn danh: 1.244 bài, 20.468 bình luận | `data/processed/`, rồi `dev.csv`, `corpus.csv` của 3 bài toán |

Muốn dựng lại từ đầu thì chạy `python scripts/fetch_data.py`, sau đó `python -m src.classifiers.data`.

**Model: chưa tải sẵn.** Thư viện `transformers` tự tải model từ Hugging Face ở lần chạy đầu, chỉ cần có mạng. Trên Kaggle phải bật Internet trong phần Settings.

| Model | Dùng cho | Kích thước | Ghi chú |
|---|---|---|---|
| `vinai/phobert-base-v2` | Clickbait, thiếu văn minh (fine-tune) | Khoảng 135 triệu tham số | Chỉ có file `.bin`, nên chỉ chạy được trên Kaggle (máy cá nhân dùng torch 2.2 không nạp được). Văn bản nên được tách từ trước (`pip install pyvi`). Fine-tune xong lưu bằng `save_pretrained(..., safe_serialization=True)` thì máy cá nhân nạp được |
| `5CD-AI/Vietnamese-Sentiment-visobert` | Cảm xúc (dùng ngay, hoặc fine-tune thêm) | Khoảng 98 triệu tham số | Có `.safetensors`, chạy được cả trên máy cá nhân. Đầu ra 3 lớp: kiểm tra thứ tự nhãn trong `config.id2label` trước khi dùng |
| `Qwen/Qwen2.5-7B-Instruct` | LLM zero-shot, few-shot, CoT cho cả 3 bài toán | 7 tỷ tham số, khoảng 5 GB khi nạp 4-bit | Chỉ chạy trên Kaggle GPU (T4 x2 hoặc P100), cần `bitsandbytes`. Notebook mẫu: `notebooks/clickbait_qwen.ipynb` |
| `Qwen/Qwen2.5-1.5B-Instruct` | Thử nhanh prompt trước khi chạy bản 7B | 1,5 tỷ tham số | Chạy được trên Kaggle, rất chậm trên CPU |

**Cách chạy trên Kaggle:**
1. Tạo một Kaggle dataset (private) từ thư mục `data/tasks/<bai-toan>/`.
2. Tạo notebook, chọn GPU, bật Internet, thêm dataset vừa tạo.
3. Chép code từ notebook mẫu và đổi tên bài toán.
4. Tải file kết quả từ `/kaggle/working` về `results/<bai-toan>/`.
5. Ghi lại phiên bản GPU, thời gian chạy và seed vào file ghi chép.

## 4. Quy trình lặp

Lặp các bước sau cho tới khi nhãn `dev` ổn định:

1. **Chạy model để chấm điểm `dev`.** Trong các lượt sửa, chỉ cần các model rẻ: mốc, TF-IDF và model có sẵn không cần huấn luyện (ví dụ ViSoBERT sentiment). PhoBERT, Qwen và ablation để tới khi nhãn đã ổn định. Riêng lượt đầu tiên nên dùng model **không** học từ `dev`, vì model đã học nhãn cũ sẽ khó chỉ ra chỗ nhãn cũ sai.
2. **Chọn mẫu để kiểm tra.** Dùng model tốt nhất hiện có:
   - Với mỗi lớp, lấy 10 mẫu model tin chắc nhất là thuộc lớp đó.
   - Lấy thêm 10 mẫu model phân vân nhất:
     - Hai lớp: điểm gần ngưỡng nhất.
     - Ba lớp: chênh lệch nhỏ nhất giữa hai lớp có xác suất cao nhất.
   - Không lấy lại mẫu đã kiểm tra ở lượt trước.
3. **Kiểm tra bằng tay.**
   - Đọc nội dung trước, tự quyết nhãn theo định nghĩa ở mục 6, rồi mới so với cột `label`.
   - **Thực tế đã làm (chốt 25/11/2026):** ở cả 3 bài toán, các lượt kiểm tra đều do LLM (Claude) quyết thay người, theo định nghĩa ở mục 6. Vì vậy nhãn `dev` là nhãn LLM đã được LLM rà lại, và mọi điểm số đo mức khớp với nhãn LLM. File ghi chép của từng bài phải nói rõ điều này. Chất lượng nhãn được đo bằng độ đồng thuận với một người gán độc lập (mục 8). Người gán ở mục 8 không được dùng công cụ AI.
   - Sai thì sửa thẳng cột `label` trong `dev.csv`. Không sửa, xóa hay đổi thứ tự cột và dòng khác. Lưu lại dạng CSV UTF-8.
4. **Commit mỗi lượt một lần,** ví dụ: `dev: luot 2, kiem tra 40 mau, sua 6 nhan`.
5. **Chạy lại model trên nhãn mới** và quay về bước 2.

**Khi định nghĩa còn mơ hồ:**
- Gặp trường hợp mục 6 không nói rõ thì ghi lại `id` và cách đã quyết vào file ghi chép.
- Nếu cần đổi hoặc thêm quy tắc thì báo cả nhóm và sửa mục 6 của file này. Sau đó rà lại các mẫu cùng kiểu đã sửa trước đó, để nhãn nhất quán.
- Dũng (cảm xúc) và Tấn (thiếu văn minh) dùng chung 1.100 bình luận nhưng mỗi người chỉ sửa file `dev.csv` của bài toán mình.

**Khi nào dừng:** khi một lượt gần như không phải sửa nhãn nào (khoảng 1 đến 2 trên 30 đến 40 mẫu), hoặc khi điểm của model không còn tăng.

**Bước cuối:**
1. Chọn model cuối, ưu tiên model đơn giản hơn nếu chênh lệch không đáng kể.
2. Huấn luyện lại model đó trên toàn bộ dữ liệu được phép dùng.
3. Gán nhãn cho `corpus.csv`, lưu vào `data/tasks/<bai-toan>/corpus_labels.csv` với đúng 3 cột `id`, `label`, `score`. `score` là xác suất lớp `1` (clickbait, thiếu văn minh) hoặc xác suất của nhãn được gán (cảm xúc), làm tròn 4 chữ số (`common.round_scores`).
4. Với các mẫu nằm trong `dev`, giữ nhãn `dev` ở cột `label` (ghép bằng `common.dev_to_corpus_ids`). `score` vẫn là điểm của model.

Mẫu code chọn mẫu kiểm tra cho bài toán hai lớp:

```python
import pandas as pd
from src.classifiers.common import load_dev

dev = load_dev("incivility")
score = pd.Series(scores, index=dev["id"])               # xác suất lớp 1 của model tốt nhất trên dev
checked = set()                                         # id đã kiểm tra ở các lượt trước
s = score.drop(list(checked & set(score.index)))
t = s.quantile(1 - (dev["label"] == "1").mean())        # ngưỡng cho tỷ lệ dự đoán dương bằng tỷ lệ nhãn
pick = set(s.nlargest(10).index) | set(s.nsmallest(10).index) | set((s - t).abs().nsmallest(10).index)
```

## 5. Việc của từng bài toán

Mỗi bài toán so sánh các mức model từ đơn giản đến phức tạp, rồi chạy ablation. Ablation nghĩa là thay đúng một thành phần mỗi lần để xem thành phần đó có ích không. Model lớn chạy trên Kaggle GPU (xem mục 3).

**Quy ước chung để ba bài toán so sánh được với nhau:**
- **Seed:** cố định `42`. Model có yếu tố ngẫu nhiên (fine-tune, Random Forest) chạy 3 seed, báo cáo trung bình và độ lệch chuẩn.
- **Lưu kết quả:** dự đoán trên `dev` của mỗi cấu hình lưu vào `results/<bai-toan>/dev_scores_<cau-hinh>.csv`.
  - Cột: `id`, `pred`, và điểm của model. Hai lớp: `score` là xác suất lớp 1. Ba lớp: `p_neg`, `p_neu`, `p_pos`.
  - Model có học từ `dev` thì lưu dự đoán ngoài fold.
- **Few-shot không được rò rỉ:** ví dụ few-shot không được chứa chính mẫu đang chấm.
  - Lấy ví dụ từ `train.csv` nếu có.
  - Riêng cảm xúc (không có `train.csv`): lấy ví dụ từ các fold khác của `dev`, hoặc chọn sẵn một bộ ví dụ cố định rồi loại các mẫu đó khỏi phần chấm điểm.
- **Không dùng chung ngưỡng:** ngưỡng và siêu tham số không được chọn trên chính phần dùng để báo cáo điểm.

### Clickbait
Code có sẵn:
- `python -m src.classifiers.clickbait ladder`: mốc và mô hình cổ điển, khoảng 10 phút trên CPU.
- `notebooks/clickbait_phobert.ipynb` và `notebooks/clickbait_qwen.ipynb`: chạy trên Kaggle.
- `python -m src.classifiers.clickbait collect`: gộp kết quả.

| Mức | Model |
|---|---|
| Mốc | Lớp đa số; luật từ khóa (dấu hỏi, dấu chấm than, "sốc", "bất ngờ", "lý do"...) |
| Cổ điển | TF-IDF (n-gram từ và ký tự) với Logistic Regression, Linear SVM, Naive Bayes, học trên `train.csv` |
| Học sâu | Fine-tune `vinai/phobert-base-v2` trên `train.csv` |
| LLM | `Qwen/Qwen2.5-7B-Instruct` 4-bit: zero-shot, few-shot, few-shot có suy luận từng bước (CoT) |

**Ablation:**
- Đầu vào: chỉ tiêu đề, hoặc tiêu đề và sapo.
- Dữ liệu huấn luyện: toàn bộ ViClickbait; chỉ phần VnExpress của ViClickbait; ViClickbait cộng `dev` (kiểm định chéo).
- PhoBERT: có hoặc không tách từ (`pyvi`).

**Độ đo:** F1 của lớp clickbait, PR-AUC, kèm khoảng tin cậy 95%.

Kết quả chạy thử trước khi chốt `dev` (500 tiêu đề lấy ngẫu nhiên, nhãn chưa sửa):
- TF-IDF với Logistic Regression: F1 khoảng 0,34, PR-AUC khoảng 0,32 (mốc ngẫu nhiên 0,10).
- Thêm sapo làm kết quả kém đi.
- Cần chạy lại trên `dev` hiện tại (750 tiêu đề).

### Cảm xúc
Viết `src/classifiers/sentiment.py` theo khuôn của `clickbait.py`.

| Mức | Model |
|---|---|
| Mốc | Lớp đa số; từ điển cảm xúc không giám sát SO-PMI: tính PMI của từ với vài từ hạt giống tích cực, tiêu cực trên `corpus.csv`, điểm bình luận là tổng điểm các từ |
| Cổ điển | TF-IDF với Logistic Regression, học trên `dev` (kiểm định chéo) |
| Học sâu | `5CD-AI/Vietnamese-Sentiment-visobert` dùng sẵn, không huấn luyện; cùng model đó fine-tune thêm trên `dev` (kiểm định chéo) |
| LLM | Qwen2.5-7B-Instruct: zero-shot, few-shot, CoT |

**Ablation:** có hoặc không đưa `parent_text` làm ngữ cảnh với bình luận trả lời.

**Độ đo:** macro-F1, F1 của từng lớp, kèm ma trận nhầm lẫn.

Lưu ý khi chạy trên máy cá nhân:
- `transformers` chỉ nạp được model có file `.safetensors` với `torch` 2.2. Model ViSoBERT ở trên chạy được.
- Chạy cả 20.468 bình luận trên CPU mất khoảng 2 giờ, nên dùng Kaggle.

### Thiếu văn minh
Viết `src/classifiers/incivility.py` theo khuôn của `clickbait.py`.

| Mức | Model |
|---|---|
| Mốc | Lớp đa số; danh sách từ miệt thị và cụm coi thường ("đọc lại bài đi", "có hiểu gì không"...) |
| Cổ điển | TF-IDF với Logistic Regression (`class_weight="balanced"`), học trên `train.csv` |
| Học sâu | Fine-tune `vinai/phobert-base-v2` |
| LLM | Qwen2.5-7B-Instruct: zero-shot, few-shot, CoT; đưa định nghĩa ở mục 6 vào prompt |

**Ablation:**
- Dữ liệu huấn luyện: chỉ UIT-ViCTSD; chỉ `dev` (kiểm định chéo); cả hai.
- Có hoặc không có `parent_text` làm ngữ cảnh.
- Ngưỡng quyết định: 0,5, hoặc ngưỡng tốt nhất theo F1 (chọn trong phần huấn luyện).

**Độ đo:** F1 của lớp thiếu văn minh, PR-AUC, kèm khoảng tin cậy 95%. Báo cáo thêm riêng cho bình luận gốc và bình luận trả lời.

**Lưu ý:**
- UIT-ViCTSD gán nhãn "toxic", định nghĩa không trùng hoàn toàn với "thiếu văn minh". Đó chính là điều ablation dữ liệu huấn luyện cần kiểm tra.
- Phần lớn mẫu dương trong `dev` là dạng coi thường ý kiến người khác. Các dạng khác ít hơn.

### Khuyến khích thử thêm
Các bảng ở trên là phần tối thiểu. Mỗi người nên thử thêm vài hướng dưới đây (hoặc hướng khác), miễn là:
- so sánh với mốc;
- ghi kết quả vào file ghi chép, kể cả khi không tốt hơn.

**Thêm mốc (baseline):**
- Đặc trưng viết tay với Logistic Regression: độ dài, dấu câu, số, chữ hoa, ngoặc kép, emoji, có `@USER`.
- Model chỉ dùng từ điển, không học.
- Đoán ngẫu nhiên theo tỷ lệ lớp.

**Thêm model:**
- `uitnlp/visobert`: BERT học trên văn bản mạng xã hội tiếng Việt, gần với bình luận. Chỉ có file `.bin`, nên chạy trên Kaggle.
- So Qwen 1.5B với Qwen 7B: model lớn hơn có đáng chi phí không.
- Fine-tune Qwen bằng LoRA trên `train.csv` hoặc `dev.csv` (kiểm định chéo).
- Gộp nhiều model (ensemble), ví dụ trung bình xác suất của TF-IDF và PhoBERT.

**Thêm ablation:**
- Đường cong học: dùng 25%, 50%, 75%, 100% nhãn `dev` khi huấn luyện, để xem cần bao nhiêu nhãn.
- Cân bằng lớp: `class_weight`, lấy mẫu lại, hay chỉ chỉnh ngưỡng.
- Tách từ hay không tách từ với TF-IDF.
- Số ví dụ few-shot (0, 4, 8, 16) và cách chọn ví dụ (ngẫu nhiên hay giống câu cần gán).
- Lệch phân phối: model học trên `train.csv` (báo khác, tỷ lệ dương khác) chuyển sang VnExpress tốt đến đâu, và chỉnh ngưỡng theo tỷ lệ của VnExpress có giúp không.
- Dùng nhãn của bài toán khác làm đặc trưng, ví dụ cảm xúc làm đặc trưng cho thiếu văn minh.

Model nào cũng nên ghi kèm thời gian chạy và phần cứng, để so được chi phí giữa các mức model.

## 6. Định nghĩa nhãn (tóm tắt)

**Clickbait (`1`):** tiêu đề có ít nhất một dấu hiệu sau.
1. Cố ý giấu thông tin chính để kích thích tò mò:
   - Đại từ mơ hồ: "người này", "điều đó", "lý do...".
   - Câu hỏi gợi tò mò mà tiêu đề không trả lời: "Vì sao...?", "Điều gì...?".
   - Danh sách không nói rõ nội dung.
   - Biệt danh thay cho tên thật.
2. Từ ngữ cảm xúc, cường điệu do người viết chọn: "gây sốc", "chấn động", "gây bão", ẩn dụ giật gân. Dấu hiệu này vẫn tính là clickbait kể cả khi sapo cho thấy tiêu đề nói đúng.
3. Bỏ sót chi tiết then chốt khiến người đọc dễ hiểu sai.

Các trường hợp gán `0`:
- Tiêu đề đủ thông tin, khách quan.
- Từ mạnh nhưng mô tả sự kiện: "bắt", "tử vong", "khởi tố".
- Trích dẫn có nêu người nói.
- Tên tác phẩm trong ngoặc kép.
- Câu hỏi chính sách hoặc hướng dẫn thật: "Có nên...?", "Làm ... thế nào?".

**Cảm xúc:** gán theo thái độ của người viết, không theo việc tin vui hay tin buồn.
- `pos`: khen, ủng hộ, vui mừng, chúc mừng.
- `neu`: hỏi thông tin, nêu sự kiện, không rõ thái độ.
- `neg`: chỉ trích, bức xúc, mỉa mai, lo lắng, buồn, chia buồn.

Trường hợp khó:
- Mỉa mai bằng từ tích cực vẫn là `neg`.
- Đồng tình với một ý kiến thì xét nội dung được đồng tình.
- Có cả khen và chê thì chọn ý chiếm ưu thế.

**Thiếu văn minh (`1`):** có ít nhất một dạng sau.
- Gọi tên xấu, lăng mạ.
- Hạ thấp hoặc quy chụp một người, một nhóm người.
- Cáo buộc nói dối, bịa đặt.
- Coi thường ý kiến hoặc cách nói của người khác: "nói như đúng rồi", "đọc lại bài đi", "có hiểu gì không".
- Mỉa mai, giễu cợt nhắm vào một người cụ thể.
- Từ tục hoặc viết tắt từ tục.

Phản đối gay gắt nhưng nhắm vào sự việc, có lý lẽ, không hạ thấp con người thì vẫn là `0`. Gọi tội phạm bằng từ trung tính ("bị cáo", "hung thủ") là `0`; dùng từ miệt thị ("súc vật", "cầm thú") là `1`.

## 7. File ghi chép (bắt buộc)

Khi xong, viết `notes/<bai-toan>.md` trên branch của mình, gồm các mục sau:

1. **Dữ liệu:** số mẫu, phân bố nhãn của `dev` trước và sau khi sửa; dữ liệu huấn luyện đã dùng.
2. **Các lượt sửa nhãn:** mỗi lượt ghi
   - số mẫu đã kiểm tra, chia theo nhóm (tin chắc từng lớp, phân vân);
   - số nhãn đã sửa và theo chiều nào (ví dụ 0 thành 1);
   - vài `id` ví dụ, lý do sửa và kiểu lỗi hay gặp;
   - mã commit tương ứng.
3. **Các model:** model, đầu vào, dữ liệu huấn luyện, siêu tham số, seed, chạy ở đâu (máy cá nhân hay Kaggle) và mất bao lâu.
4. **Kết quả:** bảng điểm của mọi model và mọi ablation, kèm khoảng tin cậy. Ghi rõ điểm tính trên nhãn của lượt nào.
5. **Model cuối:** chọn model nào, vì sao, điểm cuối cùng, tỷ lệ nhãn dự đoán trên `corpus.csv`.
6. **Phân tích lỗi:** các kiểu model hay sai, kèm ví dụ (`id`), và điểm mạnh, điểm yếu của từng mức model (độ chính xác, chi phí, cần bao nhiêu dữ liệu có nhãn).
7. **Chạy lại:** lệnh hoặc notebook cần chạy để ra lại đúng kết quả.
8. **Độ đồng thuận với người:** kết quả của mục 8 (số mẫu, tỷ lệ khớp, Cohen's kappa, các `id` lệch và nhận xét).

Không chép nội dung bình luận vào file ghi chép (repo công khai, bình luận là nội dung của người dùng): chỉ ghi `id`. Tiêu đề bài báo được trích cụm ngắn để minh họa lỗi. Mọi con số trong file ghi chép phải ra được bằng code trong repo (so sánh hai model dùng `common.paired_bootstrap`).

Kết quả thô (bảng, dự đoán) để trong `results/<bai-toan>/`. Thư mục này không đưa lên git, nên các con số quan trọng phải được chép vào file ghi chép.

## 8. Đo độ đồng thuận với người

Nhãn `dev` do LLM gán và rà lại (mục 4), nên cần biết chúng lệch bao nhiêu so với người đọc thật.

1. Người phụ trách bài toán chạy `python -m src.classifiers.agreement sample <bai-toan>`. Lệnh này ghi `data/tasks/<bai-toan>/agreement.csv`: 100 mẫu `dev` lấy ngẫu nhiên, phân tầng theo nhãn, seed 42, cột `label` để trống.
2. Một thành viên **khác** gán cột `label` bằng tay, theo định nghĩa ở mục 6, không xem `dev.csv` và không dùng công cụ AI. Đề xuất: Dũng gán clickbait, Tấn gán cảm xúc, Kiên gán thiếu văn minh.
3. Chạy `python -m src.classifiers.agreement score <bai-toan>` để tính tỷ lệ khớp và Cohen's kappa (ghi vào `results/<bai-toan>/agreement.json`). Commit `agreement.csv`, chép kết quả vào mục 8 của file ghi chép.
4. Không sửa `dev.csv` theo kết quả này: các mẫu lệch chỉ được ghi lại và phân tích. Nếu kappa thấp (dưới khoảng 0,6), báo cả nhóm trước khi dùng nhãn cho báo cáo.
