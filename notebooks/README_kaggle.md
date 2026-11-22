# Chạy notebook clickbait trên Kaggle

Hai notebook: `clickbait_phobert.ipynb` (fine-tune PhoBERT) và `clickbait_qwen.ipynb` (Qwen2.5 zero-shot, few-shot, CoT).
Cả hai chạy từ trên xuống dưới, không cần sửa code. Muốn bỏ bớt phần tốn GPU thì sửa các cờ ở ô cấu hình đầu notebook.

## 1. Tạo bộ dữ liệu trên máy

```bash
python -m src.classifiers.clickbait kaggle-bundle
```

Lệnh này ghi ra `data/kaggle/clickbait/` ba file:
- `viclickbait.csv` (id, title, sapo, source, y);
- `dev_headlines.csv` (id, title, sapo, y);
- `corpus_headlines.csv` (id, title, sapo; 1.244 dòng).

## 2. Tạo Kaggle dataset (private)

1. Kaggle → Datasets → New Dataset → kéo thả cả ba file trong `data/kaggle/clickbait/`.
2. Đặt tên, ví dụ `visim-clickbait`, chọn **Private** → Create.
3. **Khi nhãn `dev` thay đổi** (sau mỗi lượt sửa, nhất là lượt cuối): chạy lại `kaggle-bundle`, vào dataset → New Version → tải lên lại ba file. Notebook dùng nhãn `dev` để chọn cấu hình tốt nhất và cho tập huấn luyện `ViClickbait + dev`, nên phải chạy lại với nhãn đã chốt.

## 3. Tạo notebook và chạy

Làm cho từng notebook:
1. Kaggle → Code → New Notebook → File → Import Notebook → chọn file `.ipynb` trong `notebooks/`.
2. Settings (thanh bên phải):
   - Accelerator: **GPU T4 x2** (hoặc P100). Notebook chỉ dùng một GPU.
   - Internet: **On** (để tải model từ Hugging Face và `pip install`).
3. Add Input → Your Datasets → chọn dataset ở bước 2. Notebook tự tìm file trong `/kaggle/input/`.
4. Chạy: **Save Version → Save & Run All (Commit)**. Cách này chạy nền, đóng trình duyệt vẫn chạy, kết quả nằm ở tab Output của version. (Run All trong phiên tương tác cũng được, nhưng phiên bị ngắt nếu để lâu không thao tác.)

## 4. Thời gian chạy ước tính (T4, cờ mặc định)

Số liệu là ước lượng, thời gian thật được ghi trong file JSON.

| Notebook | Phần | Cờ | Ước tính |
|---|---|---|---|
| PhoBERT | Tách từ hay không, chỉ tiêu đề, 3 seed (6 lần huấn luyện) | luôn chạy | 8–12 phút |
| | Tiêu đề + sapo, 3 seed | `RUN_SAPO` (`SAPO_SEGMENTATION="both"` thì gấp đôi) | 6–10 phút |
| | ViClickbait + dev, 5 fold, seed 42 | `RUN_OOF` | 6–12 phút |
| | Model cuối, gán nhãn corpus, lưu model | `RUN_FINAL` | 2–4 phút |
| | **Tổng** | | **khoảng 25–40 phút** (P100 chậm hơn một chút) |
| Qwen | Tải và nạp Qwen2.5-7B 4-bit | luôn chạy | 5–10 phút |
| | 7B: zero, few, few4, few16, fewsim8 | `FEWSHOT_K`, `RUN_FEWSIM` | 20–35 phút |
| | 7B: CoT (sinh 200 token mỗi tiêu đề) | `RUN_COT` | 45–75 phút |
| | 7B: gán nhãn corpus bằng `few` | `RUN_FINAL` | 5–10 phút |
| | Qwen2.5-1.5B, mọi chế độ | `RUN_SMALL` | 15–30 phút |
| | **Tổng** | | **khoảng 1,5–2,5 giờ** |

Cả hai đều dưới giới hạn 12 giờ một phiên; tổng khoảng 2–3 giờ GPU trên hạn mức 30 giờ/tuần.
Thiếu hạn mức thì tắt `RUN_COT` và `RUN_SMALL` trước (tiết kiệm khoảng 1–1,5 giờ).

## 5. Tải kết quả về

Từ tab Output (hoặc `/kaggle/working`), tải về và đặt vào `results/clickbait/kaggle/`:

**PhoBERT**
- `dev_scores_phobert_seg.csv`, `dev_scores_phobert_noseg.csv`, `dev_scores_phobert_<seg|noseg>_sapo.csv`: điểm trung bình 3 seed, `pred` ở ngưỡng 0,5.
- `dev_scores_<config>_tuned.csv`: cùng điểm, `pred` ở ngưỡng tối ưu F1 chọn trên 10% ViClickbait để riêng (không bao giờ chọn trên dev).
- `dev_scores_<config>_seed42.csv`, `_seed43`, `_seed44`: điểm từng seed, để `collect` tính lại trung bình và độ lệch chuẩn theo nhãn dev mới nhất.
- `dev_scores_phobert_vc_plus_dev_oof.csv` (+ `_tuned`): ViClickbait + dev, dự đoán ngoài fold.
- `holdout_scores_<config>.csv`: điểm trên phần ViClickbait để riêng (để kiểm tra lại ngưỡng).
- `corpus_scores_phobert.csv` (id, score, pred, pred_tuned) và `corpus_scores_phobert_tuned.csv` (pred ở ngưỡng đã chỉnh).
- `phobert_runs.json`: GPU, phiên bản torch/transformers, siêu tham số, seed, thời gian từng lần chạy, điểm từng seed.
- Thư mục `phobert_clickbait_final/` (khoảng 540 MB, safetensors; `clickbait_meta.json` ghi có tách từ hay không và ngưỡng). Chỉ cần nếu muốn chạy lại model trên máy.

**Qwen**
- `dev_scores_qwen_zero.csv`, `_few.csv`, `_cot.csv`, `_few0/_few4/_few8/_few16.csv` (`few0` = `zero`, `few8` = `few`), `_fewsim8.csv`; bản 1.5B có tiền tố `qwen15_`.
- `qwen_cot_outputs.jsonl`, `qwen15_cot_outputs.jsonl`: lời giải thích của CoT, dùng cho phân tích lỗi.
- `corpus_scores_qwen.csv`.
- `qwen_runs.json`: GPU, phiên bản, prompt, thời gian từng chế độ và từng model.

Sau đó chạy trên máy:

```bash
python -m src.classifiers.clickbait collect
# nếu chọn model Kaggle làm model cuối:
python -m src.classifiers.clickbait final --from-file results/clickbait/kaggle/corpus_scores_<config>.csv
```

## 6. Ghi vào `notes/clickbait.md`

- GPU, thời gian chạy, seed: lấy từ `phobert_runs.json` và `qwen_runs.json` (khóa `env`, `seconds`, `seed`/`seeds`).
- Lần chạy ViClickbait + dev chỉ dùng seed 42 (ghi trong JSON, khóa `oof`).
- Model PhoBERT cuối học trên 90% ViClickbait + toàn bộ dev; 10% còn lại để chọn ngưỡng.
- Ghi rõ điểm tính trên nhãn dev của lượt nào (version của Kaggle dataset).
