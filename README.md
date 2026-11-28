# visim

**Real or Simulated Readers? Mining Vietnamese News Comments and Evaluating LLM-Generated Discussions**

Khai phá bình luận báo VnExpress (5 chuyên mục) theo ba nhánh: nội dung (tiêu đề clickbait), cấu trúc (cây trả lời, đồ thị người dùng), cảm xúc (cảm xúc và mức thiếu văn minh). Sau đó so sánh luồng bình luận thật với luồng do LLM (Qwen2.5-7B-Instruct) mô phỏng.

## Dữ liệu

Dữ liệu chốt ngày 13/10/2026: 1.244 bài, 20.468 bình luận. Mã người dùng được băm HMAC-SHA256, tên hiển thị và ảnh đại diện không được lưu, số điện thoại và email bị che.

| Thư mục | Nội dung |
|---|---|
| `data/processed/` | `articles`, `comments`, `threads` (chỉ số cây), `users`, `reply_centrality`, `thread_sentiment` (dạng parquet), các file tóm tắt `*.json`, `edge_check*.csv` (kiểm tra tay cây đã khôi phục) |
| `data/tasks/<clickbait\|sentiment\|incivility>/` | `dev.csv` (nhãn đánh giá), `train.csv` (dữ liệu công khai, nếu có), `corpus_labels.csv` (`id,label,score`: nhãn của model cuối cho toàn bộ dữ liệu) |
| `data/tasks/clickbait/corpus.csv`, `data/tasks/comments_corpus.csv` | Đầu vào để gán nhãn: toàn bộ bài báo; toàn bộ bình luận (dùng chung cho cảm xúc và thiếu văn minh) |
| `data/external/` | ViClickbait-2025 và UIT-ViCTSD (tải bằng `scripts/fetch_data.py`) |
| `data/simulation/` | 100 bài mô phỏng, ví dụ văn phong, mốc truy hồi, mốc trung bình chuyên mục (chỉ có id và số) |

Nhãn `dev` do LLM gán theo một hướng dẫn viết sẵn, rồi được rà lại qua các vòng sửa cũng có LLM hỗ trợ. Lịch sử thay đổi xem bằng git.

## Mã nguồn

| Gói | Vai trò |
|---|---|
| `src/crawler/` | Thu thập bài và bình luận: giới hạn tốc độ, tuân thủ robots.txt, dừng khi gặp 403/429. Đã dừng thu thập |
| `src/processing/` | Ẩn danh, chốt dữ liệu, cây hai cấp và cây khôi phục từ thẻ nhắc tên, kiểm tra cạnh bằng tay |
| `src/features/` | Chỉ số cấu trúc, đồ thị người dùng–bài báo (Louvain), đồ thị trả lời (PageRank, HITS) |
| `src/classifiers/` | Thang model cho clickbait, cảm xúc, thiếu văn minh; độ đo và bootstrap dùng chung (`common.py`) |
| `src/analysis/` | Chỉ số cảm xúc theo luồng; so luồng bài clickbait với bài thường; dự đoán tranh cãi sớm |
| `src/simulation/` | Chọn bài, ví dụ văn phong, các mốc so sánh, prompt, kiểm tra đầu ra, sinh luồng bằng vLLM (`generate.py`) |
| `notebooks/` | Notebook GPU (PhoBERT, ViSoBERT, Qwen) của ba bài toán phân loại |

## Cài đặt và chạy

Python 3.9 trở lên: `pip install -r requirements.txt`. Mọi lệnh chạy từ thư mục gốc của repo.

```bash
python -m src.features.structure            # cây và chỉ số cấu trúc
python -m src.features.user_graph           # đồ thị người dùng, Louvain
python -m src.features.reply_graph          # đồ thị trả lời, độ đo trung tâm
python scripts/fetch_data.py                # tải dữ liệu công khai
python -m src.classifiers.data              # dựng train.csv và các corpus
python -m src.classifiers.<clickbait|sentiment|incivility> ladder
python -m src.analysis.thread_sentiment     # p_neg, p_uncivil, polarization theo luồng
python -m src.analysis.clickbait_effect     # -> results/clickbait_effect/
python -m src.analysis.controversy          # -> results/controversy/ (--skip-ablations, --label-after-k)
python -m src.simulation.sample             # -> data/simulation/
python -m src.simulation.prompts preview ID A|B|C
```

Sinh luồng mô phỏng (cần GPU, vLLM và Qwen2.5-7B-Instruct, chạy trên server):

```bash
python -m src.simulation.generate download                 # tải model vào cache Hugging Face
python -m src.simulation.generate run --out data/simulation/generated/main   # 100 bài x A, B, C
python -m src.simulation.label sentiment     # huấn luyện lại ViSoBERT cuối, chấm bình luận thật và mô phỏng
python -m src.simulation.label incivility    # chạy được trên CPU
```

Kết quả ghi vào `results/` (không đưa lên git). Khóa băm nằm trong `secrets/` và không bao giờ được đưa lên git.

## Giấy phép

MIT (xem `LICENSE`). Bình luận vốn công khai trên VnExpress và đã được ẩn danh trước khi lưu.
