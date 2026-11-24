# Ghi chép: bài toán Clickbait

Phụ trách: Kiên. Branch `clickbait-kien`. Mọi số liệu dưới đây chép từ `results/clickbait/` (không đưa lên git).

> **Trạng thái:** đã chạy đủ mọi mức model: mốc, cổ điển, học sâu (PhoBERT), LLM (Qwen). Đã có ablation, model cuối và `corpus_labels.csv`.
> PhoBERT và Qwen chạy trên Kaggle; cách chạy, thời gian ước tính và danh sách file kết quả nằm ở ô đầu của mỗi notebook (`notebooks/clickbait_phobert.ipynb`, `notebooks/clickbait_qwen.ipynb`).

## 1. Dữ liệu

| Tập | Số mẫu | Nhãn dương | Ghi chú |
|---|---|---|---|
| `dev.csv` trước khi sửa | 750 | 96 (12,8%) | Nhãn do LLM (Claude) gán theo định nghĩa mục 6 của TASK.md |
| `dev.csv` sau khi sửa | 750 | **89 (11,9%)** | 4 lượt, 359 mẫu được kiểm tra, 15 nhãn đổi (4 nhãn 0→1, 11 nhãn 1→0) |
| Phần ngẫu nhiên của dev (id `001`–`500`) | 500 | 10,0% (sau sửa) | |
| Phần chọn thêm (id `H001`–`H250`) | 250 | 18,8% (trước sửa) | Chọn thêm để tăng mẫu dương, nên không đại diện cho toàn bộ dữ liệu |
| `train.csv` (ViClickbait-2025) | 3.414 | 31,2% | 8 báo; phần VnExpress có tỷ lệ dương 14,5% |
| `corpus.csv` | 1.244 | | Cả 750 tiêu đề dev đều nằm trong corpus, nhưng id dev khác id corpus, nên phải ghép theo `title` (không có tiêu đề trùng) |

Model học trên các nguồn sau: ViClickbait (toàn bộ, hoặc chỉ phần VnExpress), `dev` (kiểm định chéo), hoặc cả hai.

## 2. Các lượt sửa nhãn

**Ai quyết nhãn:** nhãn `dev` ban đầu do LLM (Claude) gán, và mọi lượt kiểm tra dưới đây cũng do LLM (Claude) quyết thay người, theo định nghĩa clickbait trong TASK.md. Vì vậy nhãn `dev` là nhãn LLM đã được LLM rà lại, và mọi điểm số trong file này đo mức khớp với nhãn LLM, không phải với người đọc. Độ đồng thuận với một người gán độc lập được đo ở mục 8.

**Cách làm:**
- Mỗi lượt chọn mẫu theo quy trình lặp trong TASK.md: 10 mẫu model chắc là 1, 10 mẫu chắc là 0, 10 mẫu phân vân (gần ngưỡng cho tỷ lệ dự đoán dương bằng tỷ lệ nhãn dương). Mẫu đã kiểm tra ở lượt trước bị loại.
- Annotator (một agent Claude riêng) chỉ thấy id, tiêu đề và sapo đã xáo thứ tự, không thấy nhãn cũ, nhóm hay điểm model; gán nhãn trước rồi mới so với nhãn cũ. Mẫu bất đồng được xét lại theo định nghĩa rồi chốt.
- Nhãn sửa thẳng ở cột `label` của `dev.csv`, mỗi lượt một commit; lịch sử thay đổi xem bằng git (commit trong bảng dưới).

| Lượt | Model chọn mẫu | Đã kiểm tra | Sửa | Chiều | Commit |
|---|---|---|---|---|---|
| 1 | TF-IDF + SVM học trên ViClickbait (không học từ dev) | 30 (10 chắc 1, 10 chắc 0, 10 phân vân) | 1 | 0→1: 1 (phân vân) | `aa0c71f` |
| 2 | TF-IDF + SVM, ViClickbait + dev (oof) | 30 (10/10/10) | 0 | | `acc4bfd` |
| 3 (quét thêm) | Không dùng model: toàn bộ mẫu dương còn lại và nhóm H chưa kiểm tra | 269 (79 dương, 190 nhóm H) | 14 | 1→0: 11 (đều trong nhóm dương); 0→1: 3 (nhóm H) | `953294c` |
| 4 | TF-IDF + SVM, ViClickbait + dev (oof), trên nhãn sau lượt 3 | 30 (10/10/10) | 0 | | `ac30c99` |

**Ví dụ và các kiểu lỗi hay gặp:**
- **Biệt danh thay tên thật bị bỏ sót (0→1):**
  - H101 *'Mẹ đẻ' Tilly Norwood: ...*: không nêu tên nhà sáng lập.
  - H035 *Tài tử 'Tế Công' qua đời*: tên vai thay cho tên thật Du Bổn Xương.
- **Danh sách không rõ nội dung (0→1):** H037 *Những tình tiết có thật trong 'Trại buôn người'*.
- **Cường điệu (0→1):** H146 *... gây rung chuyển nước Pháp*.
- **Từ trong ngoặc hoặc biệt danh đi kèm tên thật, hoặc sự việc đã nêu đủ, nhưng bị gán 1 (1→0).** Đây là lỗi phổ biến nhất của nhãn LLM ban đầu:
  - 131 *'nam thần' Nadech Kugimiya*;
  - H033 *'chú rể' Bảo Định*;
  - 120 *phong cách 'quả đào mọng'*;
  - 308 *... để dọa khách, 'moi' tiền*;
  - 023 *... để 'câu view'*;
  - 454 *cắt cảnh 'nóng'*;
  - 028, H231: người không nêu tên nhưng sự việc nêu rõ;
  - 476 *Mưa trắng trời ...*;
  - 030, H144.

**Ca mơ hồ (mục 6 chưa nói rõ) và cách đã quyết:**

| Kiểu | Quyết định | Ví dụ |
|---|---|---|
| "gây sốt" | 1, như "gây bão" | H116, 439 |
| "gây chú ý", "gây tranh luận", "thổi bùng/châm ngòi bất bình, tranh luận" | 0, vì là cụm báo chí quen dùng | 244, 060, 236, H174, H221 |
| Biệt danh hoặc mô tả **thay cho** người không nêu tên | 1 | H101, H035, 466 'Cẩu tặc', 364, 170 |
| Biệt danh **đi kèm** tên thật | 0 | H048 'Đường Tăng' La Gia Anh, 230, 131, Tuấn 'Thần đèn' |
| Biệt danh giang hồ mạng công khai | 0 | 126, H127, H168, 189 |
| Nghệ danh mặt nạ chính thức, tên thật chưa công bố | 0 | H162, H245 |
| Câu hỏi "Vì sao / Điều gì / Những … nào / có gì / bao nhiêu" mang tính gợi tò mò | 1 | 052, H243, H205, H214, H063, 259 |
| Câu hỏi cơ chế, dự báo, dịch vụ hay pháp lý thật ("thế nào / ra sao / khi nào / bao nhiêu tiền phí / có quyền … không") | 0 | 343, H182, H241, 137, H009, 351, H081 |
| Trích dẫn có người nói được nêu bằng chức danh cụ thể | 0 | H177 "Cơ phó flydubai" |

Đề xuất bổ sung các quy ước này vào mục 6 của TASK.md. Cần báo nhóm trước khi sửa.

**Điều kiện dừng:** lượt 2 và lượt 4 (chọn mẫu theo model) đều không phải sửa nhãn nào. PR-AUC của model chọn mẫu không đổi qua các lượt (0,386 → 0,393 → 0,393).

**Hạn chế:**
- Annotator và người gán nhãn ban đầu đều là Claude, nên có thể cùng thiên lệch. Tỷ lệ đồng ý ở các lượt 1, 2 và 4 là 59/60 trên các mẫu chọn theo model. Lượt quét 3 tìm ra nhiều lỗi hơn (25/269 bất đồng).
- Lượt 3 quét toàn bộ mẫu dương và toàn bộ nhóm H, nhưng không quét mẫu âm của phần ngẫu nhiên: 391 trong 458 mẫu nhãn 0 thuộc phần `001`–`500` chưa được kiểm tra lần nào (chỉ các lượt 1, 2, 4 chạm tới một ít). Vì vậy các lần sửa nghiêng về chiều 1→0: lỗi 1→0 được tìm có hệ thống, còn lỗi 0→1 trong phần ngẫu nhiên có thể còn sót. Con số 391 tính từ danh sách id của các lượt trong lịch sử git (commit `ac30c99`).

## 3. Các model

Mọi model dùng seed 42. Model cổ điển chạy trên máy cá nhân: macOS 26.1, Apple M1 (8 lõi), RAM 8 GB, Python 3.9.25, scikit-learn 1.6.1.

| Mức | Model | Đầu vào | Huấn luyện | Siêu tham số | Thời gian |
|---|---|---|---|---|---|
| Mốc | Lớp đa số; luật từ khóa (30 regex, ≥1 khớp thì gán 1, không tinh chỉnh trên dev) | title | | | < 1 s |
| Mốc | Đoán ngẫu nhiên theo tỷ lệ ViClickbait (31%) | | | | < 1 s |
| Mốc | Đặc trưng viết tay + LR: độ dài ký tự và từ, `?`, `!`, số chữ số, dấu ngoặc kép, `:`, tỷ lệ chữ hoa, số từ khóa khớp; có StandardScaler | title | ViClickbait | C ∈ {0,01; 0,1; 1; 10} chọn bằng CV 5 phần trên train → C=10 | 3 s |
| Cổ điển | TF-IDF n-gram từ (1–2) cộng ký tự `char_wb` (2–5), sublinear; LR, Linear SVM, Complement NB | title / title+sapo | ViClickbait | GridSearch CV 5 phần trên train, chấm theo F1: LR C=10, SVM C=1, NB α=0,1 | toàn bộ `ladder` khoảng 2 phút trên M1 (21 phút trên MacBook Intel i3 2 lõi, chạy lại ngày 25/11/2026) |
| Cổ điển (trong miền) | TF-IDF + LR; đặc trưng viết tay + LR; **TF-IDF kết hợp đặc trưng viết tay + LR** | title | dev (oof), hoặc ViClickbait + dev (oof) | CV lồng: vòng ngoài 5 phần trên dev; vòng trong 5 phần trên phần dev của fold huấn luyện, chọn C ∈ {0,3; 1; 3; 10; 30} theo PR-AUC và chọn ngưỡng tối ưu F1 | 3–12 s (chỉ dev), 27–200 s (cả ViClickbait) trên M1; toàn bộ `in-domain` khoảng 6 phút trên M1, 14 phút trên Intel i3 |
| Học sâu | `vinai/phobert-base-v2` fine-tune: 4 epoch, lr 2e-5, batch 32, 3 seed (42, 43, 44); có hoặc không tách từ; title hoặc title+sapo | | ViClickbait (90%, 10% giữ lại để chọn ngưỡng); ViClickbait + dev (oof, chỉ seed 42) | warmup 10%, weight decay 0,01, fp16, cắt gradient 1,0; max_len 64 (title) / 160 (title+sapo) | Kaggle, 1 GPU Tesla T4; torch 2.11 (cu130), transformers 5.20.2. Mỗi cấu hình 3 seed: 127–133 s (title), 247 s (title+sapo); oof 250 s; model cuối 53 s; tổng 13,5 phút |
| LLM | `Qwen2.5-7B-Instruct` 4-bit và `1.5B`: zero-shot, few-shot k ∈ {0, 4, 8, 16} lấy từ phần VnExpress của ViClickbait, few-shot theo độ tương tự, CoT | title | không học | Giải mã tham lam; score = P("1") / (P("0") + P("1")) ở token nhãn; ngưỡng 0,5; CoT sinh tối đa 200 token; batch 8 | Kaggle, 1 GPU Tesla T4; torch 2.11 (cu130), transformers 5.20.2, bitsandbytes 0.50.2. 7B: nạp model 116 s, chiếm 5,9 GB GPU; mỗi chế độ không CoT 3,5–8 phút, CoT 28 phút, chấm corpus khoảng 10 phút. 1.5B: 0,8–3 phút mỗi chế độ |

## 4. Kết quả

Bootstrap 1.000 lần, khoảng tin cậy 95%. Mọi điểm tính trên nhãn sau lượt 4 (nhãn cuối, 89 dương), trừ bảng 4.0. Model không học từ dev được chấm trên toàn bộ dev. Model có học từ dev được chấm out-of-fold (`StratifiedKFold(5, shuffle=True, random_state=42)`).

### 4.0 Điểm ban đầu (nhãn gốc, lượt 0)

| Model | F1 | PR-AUC |
|---|---|---|
| Luật từ khóa | 0,366 | 0,278 |
| TF-IDF+LR, ViClickbait | 0,364 | 0,317 |
| TF-IDF+SVM, ViClickbait | 0,340 | 0,321 |
| TF-IDF+SVM, ViClickbait + dev (oof) | 0,402 | 0,386 |

### 4.1 Mốc và cổ điển (học trên ViClickbait)

| Model | Đầu vào | F1 [CI95] | Precision | Recall | PR-AUC [CI95] | Tỷ lệ dự đoán dương |
|---|---|---|---|---|---|---|
| Lớp đa số | | 0,000 | 0 | 0 | 0,119 [0,097–0,141] | 0 |
| Đoán ngẫu nhiên | | 0,203 [0,150–0,261] | 0,140 | 0,371 | 0,107 [0,085–0,138] | 0,315 |
| Luật từ khóa | title | 0,382 [0,282–0,481] | 0,441 | 0,337 | 0,284 [0,203–0,370] | 0,091 |
| Đặc trưng viết tay + LR | title | 0,385 [0,308–0,454] | 0,290 | 0,573 | **0,434** [0,332–0,534] | 0,235 |
| TF-IDF+LR | title | 0,353 [0,284–0,414] | 0,244 | 0,640 | 0,312 [0,231–0,399] | 0,312 |
| TF-IDF+SVM | title | 0,341 [0,272–0,405] | 0,234 | 0,629 | 0,313 [0,232–0,399] | 0,319 |
| TF-IDF+NB | title | 0,295 [0,236–0,349] | 0,188 | 0,685 | 0,226 [0,172–0,310] | 0,432 |
| TF-IDF+LR | title+sapo | 0,271 [0,199–0,340] | 0,218 | 0,360 | 0,215 [0,163–0,298] | 0,196 |
| TF-IDF+SVM | title+sapo | 0,274 [0,204–0,341] | 0,214 | 0,382 | 0,220 [0,163–0,304] | 0,212 |
| TF-IDF+NB | title+sapo | 0,254 [0,191–0,313] | 0,173 | 0,483 | 0,185 [0,144–0,253] | 0,332 |

### 4.2 Ablation (cổ điển)

| Ablation | Cấu hình | F1 [CI95] | PR-AUC [CI95] | Tỷ lệ dự đoán dương |
|---|---|---|---|---|
| Dữ liệu (SVM, title) | ViClickbait toàn bộ | 0,341 [0,272–0,405] | 0,313 [0,232–0,399] | 0,319 |
| | Chỉ phần VnExpress của ViClickbait | 0,149 [0,069–0,225] | 0,178 [0,135–0,242] | 0,079 |
| | ViClickbait + dev (oof) | 0,398 [0,318–0,475] | 0,393 [0,301–0,497] | 0,196 |
| Dữ liệu (SVM, title+sapo) | Chỉ phần VnExpress | 0,046 [0,000–0,104] | 0,141 [0,112–0,180] | 0,056 |
| | ViClickbait + dev (oof) | 0,342 [0,259–0,427] | 0,277 [0,213–0,374] | 0,131 |
| Ngưỡng (TF-IDF+LR, ViClickbait) | 0,5 | 0,353 [0,284–0,414] | 0,312 | 0,312 |
| | Tối ưu F1 trên điểm oof của ViClickbait (t=0,33) | 0,320 [0,261–0,378] | 0,312 | 0,440 |
| | Chỉnh prior 0,312 → 0,145 (tương đương ngưỡng ≈0,73) | 0,332 [0,248–0,413] | 0,312 | 0,147 |
| Tách từ (pyvi) | Có tách từ | 0,355 [0,282–0,422] | 0,319 [0,240–0,410] | 0,280 |
| Cân bằng lớp | `class_weight="balanced"` | 0,318 [0,260–0,376] | 0,286 [0,212–0,375] | 0,443 |
| Đường cong học (ViClickbait + x% phần dev huấn luyện, oof) | 0% | 0,353 | 0,312 | 0,312 |
| | 25% | 0,354 | 0,317 | 0,243 |
| | 50% | 0,352 | 0,358 | 0,207 |
| | 75% | 0,388 | 0,360 | 0,204 |
| | 100% | 0,395 [0,312–0,464] | 0,392 [0,301–0,492] | 0,179 |

### 4.3 Model trong miền (học trên dev, CV lồng, ngưỡng chọn ở vòng trong)

| Model | Huấn luyện | F1 [CI95] | Precision | Recall | PR-AUC [CI95] | Tỷ lệ dự đoán dương |
|---|---|---|---|---|---|---|
| TF-IDF+LR | dev | 0,439 [0,345–0,521] | 0,402 | 0,483 | 0,433 [0,329–0,536] | 0,143 |
| TF-IDF+LR | ViClickbait + dev | 0,379 [0,289–0,462] | 0,356 | 0,404 | 0,386 [0,294–0,488] | 0,135 |
| Đặc trưng viết tay + LR | dev | 0,420 [0,320–0,510] | 0,413 | 0,427 | 0,450 [0,339–0,559] | 0,123 |
| Đặc trưng viết tay + LR | ViClickbait + dev | 0,380 [0,297–0,454] | 0,290 | 0,551 | 0,439 [0,335–0,544] | 0,225 |
| **TF-IDF + đặc trưng viết tay + LR** | **dev** | **0,553 [0,453–0,632]** | 0,525 | 0,584 | **0,573 [0,460–0,677]** | 0,132 |
| TF-IDF + đặc trưng viết tay + LR | ViClickbait + dev | 0,462 [0,368–0,548] | 0,425 | 0,506 | 0,438 [0,335–0,555] | 0,141 |

Ghi chú: bản TF-IDF+LR chỉ học trên dev trong `extras` dùng ngưỡng 0,5. Bản này có PR-AUC 0,468 nhưng F1 chỉ 0,273, vì chỉ 2,8% mẫu được dự đoán dương. Điều đó cho thấy việc chọn ngưỡng ở vòng trong quan trọng thế nào.

### 4.4 LLM: Qwen2.5-Instruct (Kaggle, không học, chấm trên toàn bộ dev)

Điểm tính lại bằng lệnh `collect` trên nhãn cuối. Ví dụ few-shot lấy từ phần VnExpress của ViClickbait, không lấy từ dev. Hai cấu hình few0 và few8 trùng với zero và few nên không lặp lại trong bảng.

| Model | Chế độ | F1 [CI95] | PR-AUC [CI95] | Tỷ lệ dự đoán dương | Thời gian |
|---|---|---|---|---|---|
| 7B 4-bit | zero-shot | **0,406** [0,312–0,498] | 0,338 [0,252–0,441] | 0,131 | 201 s |
| 7B | few-shot k=4 | 0,392 [0,324–0,454] | 0,327 [0,253–0,437] | 0,351 | 276 s |
| 7B | few-shot k=8 | 0,402 [0,335–0,469] | 0,334 [0,259–0,440] | 0,332 | 346 s |
| 7B | few-shot k=16 | 0,384 [0,321–0,444] | 0,335 [0,259–0,443] | 0,409 | 480 s |
| 7B | few-shot 8 ví dụ giống nhất (TF-IDF ký tự) | 0,362 [0,295–0,420] | **0,344** [0,261–0,459] | 0,389 | 342 s |
| 7B | CoT (8 ví dụ, 0,1% đầu ra không đọc được nhãn) | 0,357 [0,276–0,431] | 0,291 [0,219–0,386] | 0,217 | 1.687 s |
| 1.5B fp16 | zero-shot | 0,214 [0,179–0,250] | 0,222 [0,170–0,301] | 0,992 | 47 s |
| 1.5B | few-shot k=4 | 0,221 [0,184–0,257] | 0,304 [0,227–0,402] | 0,956 | 64 s |
| 1.5B | few-shot k=8 | 0,235 [0,196–0,273] | 0,327 [0,242–0,421] | 0,891 | 81 s |
| 1.5B | few-shot k=16 | 0,236 [0,194–0,273] | 0,243 [0,182–0,327] | 0,832 | 109 s |
| 1.5B | few-shot theo độ tương tự | 0,215 [0,179–0,251] | 0,292 [0,211–0,388] | 0,985 | 80 s |
| 1.5B | CoT | 0,322 [0,269–0,375] | 0,339 [0,250–0,436] | 0,552 | 182 s |

**Nhận xét:**
- **Qwen 7B kém hơn model cuối.** F1 tốt nhất 0,406 và PR-AUC 0,34, so với 0,553 / 0,573 của model cuối, và khoảng tin cậy PR-AUC không chồng nhau. Qwen 7B chỉ ngang mức TF-IDF học trên ViClickbait và luật từ khóa.
- **Few-shot không giúp gì.** Tăng k chỉ làm model gán dương nhiều hơn (13% → 41%). Lý do là ví dụ lấy từ ViClickbait, mà nhãn của bộ này lệch với định nghĩa của nhóm.
- **CoT kém hơn không có CoT.** Lời giải thích hay mâu thuẫn với chính nhãn model đưa ra (xem mục 6).
- **1.5B gần như gán 1 cho mọi tiêu đề** (89–99%). Nó gần như không làm được việc này, chỉ CoT là đỡ hơn. **Bản 7B đáng chi phí hơn 1.5B rõ rệt.**

### 4.5 Học sâu: PhoBERT (`vinai/phobert-base-v2`, Kaggle)

Điểm tính lại bằng lệnh `collect` trên nhãn cuối.
- Cột "điểm trung bình" dùng xác suất trung bình qua 3 seed, ngưỡng 0,5.
- Cột "mean ± std" là trung bình và độ lệch chuẩn khi chấm riêng từng seed.
- Cột "ngưỡng hold-out": ngưỡng tối ưu F1 chọn trên 10% ViClickbait được giữ lại, không chọn trên dev.

| Cấu hình | Huấn luyện | F1 [CI95] (điểm trung bình) | PR-AUC [CI95] | F1 mean ± std (3 seed) | PR-AUC mean ± std | F1 với ngưỡng hold-out | Tỷ lệ dự đoán dương |
|---|---|---|---|---|---|---|---|
| Tách từ (pyvi), title | ViClickbait | 0,500 [0,414–0,579] | 0,504 [0,397–0,611] | 0,507 ± 0,013 | 0,481 ± 0,016 | 0,486 (t=0,61) | 0,180 |
| Không tách từ, title | ViClickbait | 0,483 [0,404–0,559] | **0,514** [0,400–0,617] | 0,484 ± 0,044 | 0,488 ± 0,009 | 0,487 (t=0,51) | 0,191 |
| Không tách từ, title+sapo | ViClickbait | 0,464 [0,372–0,544] | 0,431 [0,333–0,545] | 0,463 ± 0,004 | 0,432 ± 0,012 | 0,457 (t=0,60) | 0,163 |
| Không tách từ, title | ViClickbait + dev (oof, seed 42) | **0,569 [0,477–0,647]** | **0,527 [0,413–0,655]** | (1 seed) | | 0,558 (t=0,36) | 0,144 |

**Ensemble** (trung bình xác suất, ngưỡng 0,5; lệnh `collect` tính và ghi `results/clickbait/ensemble/dev_scores_<ghép>.csv`, nên so sánh cặp được bằng `compare`):

| Ghép | F1 [CI95] | PR-AUC [CI95] | Tỷ lệ dự đoán dương |
|---|---|---|---|
| TF-IDF+LR (ViClickbait) + PhoBERT tách từ | 0,504 [0,416–0,577] | 0,458 [0,358–0,560] | 0,188 |
| TF-IDF+LR (ViClickbait) + PhoBERT ViClickbait + dev (oof) | 0,530 [0,439–0,609] | 0,498 [0,392–0,610] | 0,173 |
| **Model cuối (TF-IDF + đặc trưng viết tay, dev, oof) + PhoBERT ViClickbait + dev (oof)** (`hand_tfidf_lr_dev_oof+phobert_vc_plus_dev_oof`, thử thêm sau khi có kết quả) | **0,614 [0,510–0,701]** | **0,651 [0,538–0,756]** | 0,085 |

**Nhận xét:**
- **PhoBERT là model mạnh nhất trong các model chỉ học trên ViClickbait.** PR-AUC khoảng 0,50, so với 0,31 của TF-IDF và 0,34 của Qwen 7B, dù không thấy nhãn VnExpress nào. PhoBERT nắm được ngữ nghĩa tiêu đề tốt hơn n-gram.
- **Thêm dev vào huấn luyện giúp tăng F1** (0,48 → 0,57). PR-AUC chỉ tăng nhẹ (0,51 → 0,53).
- **Tách từ bằng pyvi không đổi kết quả:** chênh lệch nằm trong độ lệch giữa các seed.
- **Thêm sapo làm kém đi,** giống ở mức cổ điển (PR-AUC 0,51 → 0,43).
- **Ngưỡng hold-out không giúp F1.** Ngưỡng 0,5 đã cho tỷ lệ dự đoán dương gần tỷ lệ thật.
- **Ensemble của model cuối với PhoBERT oof là cấu hình tốt nhất** (F1 0,614, PR-AUC 0,651). Hai model sai ở chỗ khác nhau: model cổ điển dựa vào bề mặt tiêu đề, PhoBERT dựa vào ngữ nghĩa. Tuy vậy:
  - khoảng tin cậy vẫn chồng với model cuối;
  - cấu hình này được thử sau khi đã xem kết quả, không định trước, nên điểm có thể hơi lạc quan.

## 5. Model cuối

**Model chọn:** TF-IDF (n-gram từ và ký tự) kết hợp 9 đặc trưng viết tay, Logistic Regression, học trên 750 tiêu đề dev.

**Lý do:**
- PR-AUC cao nhất trong các model đơn lẻ: 0,573, so với 0,527 của PhoBERT. F1 ngang PhoBERT: 0,553 so với 0,569, chênh lệch nằm trong khoảng tin cậy.
- Vượt rõ mọi model cổ điển học trên ViClickbait (PR-AUC tốt nhất của nhóm này là 0,434) và Qwen (0,344).
- Model đơn giản, chạy trên CPU trong vài giây.
- Lưu ý: model này được chọn trong khoảng 30 cấu hình (chưa kể các cấu hình Kaggle) dựa trên chính điểm out-of-fold trên dev, nên điểm 0,553 / 0,573 hơi lạc quan so với điểm trên dữ liệu mới.
- Thêm ViClickbait vào dữ liệu huấn luyện lại làm kém đi, vì định nghĩa clickbait của ViClickbait (31% dương, 8 báo) khác với định nghĩa của nhóm.

**Gán nhãn corpus:** `python -m src.classifiers.clickbait final --config hand_tfidf_lr --in-domain dev`
- Huấn luyện lại trên toàn bộ dev, C=10 chọn bằng CV 5 phần trên dev.
- Ngưỡng 0,167, là ngưỡng tối ưu F1 trên điểm out-of-fold của CV 5 phần trên dev (khi nhiều ngưỡng cùng F1 thì lấy ngưỡng gần 0,5 nhất, `common.f1_threshold`). Chạy lại ngày 25/11/2026: ngưỡng không đổi, `corpus_labels.csv` giống hệt từng byte.
- PR-AUC vòng trong 0,573.

**Kết quả trên `corpus.csv` (1.244 bài):**
- 112 bài nhãn 1 (9,0%) theo cột `label`.
- 750 bài trong dev giữ nhãn sửa tay; model đồng ý với 99,1% số này vì đã học trên chúng.
- 494 bài ngoài dev: model gán 4,7% là clickbait. Phần này đã bị rút bớt mẫu dương khi chọn nhóm H, nên tỷ lệ thấp hơn mức 10% của phần dev ngẫu nhiên là hợp lý.

**So với các mức khác:**
- **Qwen** kém hơn rõ (mục 4.4).
- **PhoBERT tốt nhất** (ViClickbait + dev, oof): F1 0,569 [0,477–0,647], PR-AUC 0,527 [0,413–0,655]. Khoảng tin cậy chồng hẳn lên model cuối, F1 0,553 [0,453–0,632] và PR-AUC 0,573 [0,460–0,677].
- **Ensemble** cao hơn (0,614 / 0,651) nhưng khoảng tin cậy vẫn chồng nhau, và cấu hình này được chọn sau khi xem kết quả.

Theo TASK §4 (ưu tiên model đơn giản hơn nếu chênh lệch không đáng kể), mình **giữ model cổ điển**. Model này chạy vài giây trên CPU, không cần GPU, dễ chạy lại.

**PhoBERT bản cuối** (không tách từ, học trên 90% ViClickbait + toàn bộ dev, seed 42) chấm cả corpus vào `results/clickbait/kaggle/corpus_scores_phobert.csv`. Muốn dùng làm nhãn corpus thì chạy `final --from-file` với file này (lệnh ghi đè `corpus_labels.csv` và in tỷ lệ dương trong `results/clickbait/final.json`).

## 6. Phân tích lỗi

Phân tích này dựa trên model cuối, chấm out-of-fold trên dev.

**Dương tính giả (FP), kiểu hay gặp:**
- Từ khóa gợi tò mò nhưng mô tả đúng sự việc hoặc tên tác phẩm:
  - H071 *'Bí mật trong gương' - phá án nhờ ảo thuật* (tên tác phẩm);
  - H245, H162 "giọng ca bí ẩn" (nghệ danh mặt nạ);
  - H170 *'có bất ngờ lớn'* (trích lời ông Trump).
- Câu hỏi pháp lý thật có dấu "?": H009.
- Cụm trong ngoặc mang nghĩa mô tả: 454 cắt cảnh 'nóng', H142.
- Model gần như chỉ dựa vào bề mặt: dấu "?", dấu ngoặc, các từ "bí mật", "bí ẩn", "bất ngờ".

**Âm tính giả (FN), kiểu hay gặp.** Đây là các dấu hiệu cần hiểu ngữ nghĩa, không có từ khóa:
- ẩn dụ hoặc khung mơ hồ: 041 *Tình cảnh trơ trọi...*, 090 *Cả gia đình gặp họa vì ác tâm...*, 307 *đế chế sau song sắt*;
- biệt danh thay tên: 466 'Cẩu tặc', H101 'Mẹ đẻ', H248 'bầu' Đoan;
- danh sách hoặc người không tên: H203 *4 ứng viên nổi bật...*, H228 *Cô gái thất nghiệp đổi đời...*;
- tiêu đề cảm xúc chung chung: H129 *Niềm vui, nỗi buồn của Brad Pitt*, 268.

**Điểm mạnh và điểm yếu theo mức model:**
- **Luật từ khóa:** precision cao nhất (0,44), không cần nhãn, nhưng recall thấp (0,34).
- **Học trên ViClickbait:** đặt ngưỡng 0,5 thì dự đoán dương tới 31%, vì train có 31% dương còn dev khoảng 12%. Model gán clickbait cho cả tin bình thường như 004, 009, 018, 022.
  - Chỉnh ngưỡng hay chỉnh prior không giúp F1. Nguyên nhân chính là định nghĩa nhãn lệch nhau, không phải chỉ do lệch tỷ lệ lớp.
  - Chỉ dùng phần VnExpress của ViClickbait (khoảng 500 mẫu, 14,5% dương) còn kém hơn.
  - Thêm sapo làm kém đi ở mọi model: sapo dài, mang nội dung thật, làm loãng tín hiệu của tiêu đề.
- **Đặc trưng viết tay:** 9 đặc trưng mà PR-AUC đã ngang TF-IDF. Cấu trúc tiêu đề (dấu câu, ngoặc, độ dài) mang nhiều tín hiệu.
- **Học trong miền:** 600 nhãn dev mỗi fold đã vượt 3.414 nhãn ViClickbait. Đường cong học vẫn còn tăng ở mức 100% dev, nên thêm nhãn VnExpress sẽ còn có lợi.
  - Chi phí: vài giây trên CPU, không cần GPU.
  - Hạn chế: khoảng tin cậy rộng (khoảng ±0,1) vì chỉ có 89 mẫu dương.
- **Qwen (LLM):** không cần nhãn, nhưng tốn GPU nhất (khoảng 1,5 giờ T4 cho cả bản 7B và 1.5B) mà vẫn kém model cổ điển học trên 750 nhãn dev.
  - Qwen hiểu sai định nghĩa theo hai kiểu ngược nhau.
  - Kiểu 1, coi tin bình thường là clickbait: 025 *Tom Cruise tái hiện điệu nhảy trong 'Risky Business'* (cho là "gợi tò mò"), 028 (cho "trang sức từng tặng" là đại từ mơ hồ), 023.
  - Kiểu 2, bỏ qua dấu hiệu thật: 016 *... bị Iran 'đánh tan hoang'*, 006 *Những lời đầu tiên của tử tù...*.
  - CoT có lúc tự mâu thuẫn. Ở 015, lời giải thích nói tiêu đề "dùng từ ngữ cảm xúc" nhưng vẫn kết luận nhãn 0. Ở 025, lời giải thích nói "không phải cường điệu" nhưng lại gán nhãn 1.
- **PhoBERT (học sâu):**
  - Khi không có nhãn trong miền, đây là mức tốt nhất (PR-AUC khoảng 0,50, chỉ học trên ViClickbait). Có thêm dev thì ngang model cổ điển trong miền.
  - Chi phí vừa phải: khoảng 2 phút GPU T4 cho mỗi cấu hình 3 seed, toàn bộ notebook 13,5 phút. Nhưng bắt buộc phải có GPU, và file model nặng 540 MB.
  - Các lỗi khác với model cổ điển, nên ghép hai model cho điểm cao nhất (mục 4.5).
- **Chi phí theo mức:**

  | Mức | Thời gian chạy | Nhãn VnExpress cần |
  |---|---|---|
  | Luật từ khóa | 0 giây | 0 |
  | Cổ điển | vài giây đến vài phút trên CPU | 750 nhãn dev cho bản tốt nhất |
  | PhoBERT | khoảng 14 phút trên T4 | 0, hoặc 750 nhãn dev |
  | Qwen 7B | khoảng 1 giờ trên T4 | 0 |

  Model rẻ nhất mà đạt điểm cao nhất là model cổ điển học trên 750 nhãn dev.

## 7. Chạy lại

```bash
conda create -n visim python=3.9 && conda activate visim
pip install -r requirements.txt     # cần có pyvi (dùng cho extras, nhánh tách từ)
python -m src.classifiers.clickbait ladder          # mốc + cổ điển + ablation dữ liệu/đầu vào (~2 phút trên M1, ~20 phút trên Intel i3)
python -m src.classifiers.clickbait extras          # mốc thêm, ngưỡng, tách từ, cân bằng lớp, đường cong học
python -m src.classifiers.clickbait in-domain       # model trong miền, CV lồng (~6 phút trên M1, ~14 phút trên Intel i3)
python -m src.classifiers.clickbait final --config hand_tfidf_lr --in-domain dev   # corpus_labels.csv
python -m src.classifiers.clickbait kaggle-bundle   # đầu vào cho 2 notebook Kaggle (hướng dẫn ở ô đầu notebook)
python -m src.classifiers.clickbait collect         # chấm lại kết quả Kaggle, ensemble, gộp vào results/clickbait/ladder_full.csv
python -m src.classifiers.clickbait compare <cấu hình A> <cấu hình B>   # ΔF1, ΔPR-AUC kèm CI95 (bootstrap cặp)
```

Bảng 4.0 (nhãn gốc) và PR-AUC của model chọn mẫu qua các lượt (mục 2) ra bằng `ladder` chạy trên `dev.csv` ở commit tương ứng trong bảng các lượt sửa.

## 8. Độ đồng thuận với người

Chưa làm. Vì nhãn `dev` là nhãn LLM đã được LLM rà lại, cần đo xem chúng lệch bao nhiêu so với một người đọc thật:
1. Chạy `python -m src.classifiers.agreement sample clickbait`: ghi `data/tasks/clickbait/agreement.csv`, 100 mẫu `dev` lấy ngẫu nhiên, phân tầng theo nhãn, seed 42, cột `label` để trống.
2. Một thành viên khác (đề xuất: Dũng) gán cột `label` bằng tay theo định nghĩa clickbait trong TASK.md, không xem `dev.csv` và không dùng công cụ AI.
3. Chạy `python -m src.classifiers.agreement score clickbait`: tỷ lệ khớp và Cohen's kappa, ghi vào `results/clickbait/agreement.json`. Chép số mẫu, tỷ lệ khớp, kappa, các `id` lệch và nhận xét vào mục này.
4. Không sửa `dev.csv` theo kết quả này. Nếu kappa dưới khoảng 0,6 thì báo cả nhóm trước khi dùng nhãn cho báo cáo.
