# Ghi chép: bài toán Clickbait

Phụ trách: Kiên. Branch `clickbait-kien`. Mọi số liệu dưới đây chép từ `results/clickbait/` (không đưa lên git).

> **Trạng thái:** phần chạy trên máy cá nhân đã xong (mốc, cổ điển, ablation, model cuối, `corpus_labels.csv`).
> Mức học sâu (PhoBERT) và LLM (Qwen) **chưa chạy**: notebook đã sẵn sàng, xem [notebooks/README_kaggle.md](../notebooks/README_kaggle.md).
> Chạy xong thì điền mục 4.4 và cập nhật mục 5 nếu có model tốt hơn.

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

**Cách làm:**
- Mỗi lượt, lệnh `review` chọn mẫu theo TASK §4: 10 mẫu chắc là 1, 10 mẫu chắc là 0, 10 mẫu phân vân. Ngưỡng là quantile `1 − tỷ lệ nhãn dương`. Mẫu đã kiểm tra ở lượt trước bị loại.
- File `round_N_blind.csv` chỉ có id, tiêu đề và sapo, đã xáo thứ tự, không có nhãn cũ, nhóm hay điểm model.
- Một annotator đọc file blind và gán nhãn trước theo mục 6, sau đó mới so với nhãn cũ. Annotator là một agent Claude riêng không thấy nhãn cũ; nhóm đã thống nhất là được dùng AI để kiểm tra.
- Mẫu nào bất đồng thì được xét lại theo mục 6 rồi chốt nhãn.
- Chi tiết từng mẫu (`id, group, score, old, new, rule, reason`) nằm trong [notes/clickbait_review/round_N.csv](clickbait_review/).
- Lệnh `apply-labels` chỉ ghi cột `label` của `dev.csv`. Đã kiểm tra rằng pandas đọc rồi ghi lại file giữ nguyên từng byte.

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

**Hạn chế:** annotator và người gán nhãn ban đầu đều là Claude, nên có thể cùng thiên lệch. Tỷ lệ đồng ý ở các lượt 1, 2 và 4 là 59/60 trên các mẫu chọn theo model. Lượt quét 3 tìm ra nhiều lỗi hơn (25/269 bất đồng).

## 3. Các model

Mọi model dùng seed 42. Model cổ điển chạy trên máy cá nhân: macOS 26.1, Apple M1 (8 lõi), RAM 8 GB, Python 3.9.25, scikit-learn 1.6.1.

| Mức | Model | Đầu vào | Huấn luyện | Siêu tham số | Thời gian |
|---|---|---|---|---|---|
| Mốc | Lớp đa số; luật từ khóa (30 regex, ≥1 khớp thì gán 1, không tinh chỉnh trên dev) | title | | | < 1 s |
| Mốc | Đoán ngẫu nhiên theo tỷ lệ ViClickbait (31%) | | | | < 1 s |
| Mốc | Đặc trưng viết tay + LR: độ dài ký tự và từ, `?`, `!`, số chữ số, dấu ngoặc kép, `:`, tỷ lệ chữ hoa, số từ khóa khớp; có StandardScaler | title | ViClickbait | C ∈ {0,01; 0,1; 1; 10} chọn bằng CV 5 phần trên train → C=10 | 3 s |
| Cổ điển | TF-IDF n-gram từ (1–2) cộng ký tự `char_wb` (2–5), sublinear; LR, Linear SVM, Complement NB | title / title+sapo | ViClickbait | GridSearch CV 5 phần trên train, chấm theo F1: LR C=10, SVM C=1, NB α=0,1 | toàn bộ `ladder` khoảng 2 phút |
| Cổ điển (trong miền) | TF-IDF + LR; đặc trưng viết tay + LR; **TF-IDF kết hợp đặc trưng viết tay + LR** | title | dev (oof), hoặc ViClickbait + dev (oof) | CV lồng: vòng ngoài 5 phần trên dev; vòng trong 5 phần trên phần dev của fold huấn luyện, chọn C ∈ {0,3; 1; 3; 10; 30} theo PR-AUC và chọn ngưỡng tối ưu F1 | 3–12 s (chỉ dev), 27–200 s (cả ViClickbait) |
| Học sâu | `vinai/phobert-base-v2` fine-tune: 4 epoch, lr 2e-5, batch 32, 3 seed (42, 43, 44); có hoặc không tách từ; title hoặc title+sapo | | ViClickbait (90%, 10% giữ lại để chọn ngưỡng); ViClickbait + dev (oof) | | **chưa chạy** (Kaggle T4, ước tính 25–40 phút) |
| LLM | `Qwen2.5-7B-Instruct` 4-bit và `1.5B`: zero-shot, few-shot k ∈ {0, 4, 8, 16} lấy từ phần VnExpress của ViClickbait, few-shot theo độ tương tự, CoT | title | không học | | **chưa chạy** (ước tính 1,5–2,5 giờ) |

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

### 4.4 Học sâu và LLM (Kaggle): chưa chạy

Sau khi tải kết quả về `results/clickbait/kaggle/`, chạy `python -m src.classifiers.clickbait collect` rồi chép bảng `results/clickbait/ladder_full.csv` vào đây:
- PhoBERT tách từ / không tách từ / title+sapo: mean ± std qua 3 seed.
- PhoBERT ViClickbait + dev (oof).
- Ensemble TF-IDF + PhoBERT.
- Qwen 7B và 1.5B: zero, few k=4/8/16, few theo độ tương tự, CoT.

## 5. Model cuối

**Model chọn:** TF-IDF (n-gram từ và ký tự) kết hợp 9 đặc trưng viết tay, Logistic Regression, học trên 750 tiêu đề dev.

**Lý do:**
- Model này tốt nhất ở cả F1 và PR-AUC trong mọi model đã chạy.
- Khoảng tin cậy của nó không chồng lên các model học trên ViClickbait (PR-AUC 0,573 [0,460–0,677] so với tốt nhất 0,434).
- Model đơn giản, chạy trên CPU trong vài giây.
- Thêm ViClickbait vào dữ liệu huấn luyện lại làm kém đi, vì định nghĩa clickbait của ViClickbait (31% dương, 8 báo) khác với định nghĩa của nhóm.

**Gán nhãn corpus:** `python -m src.classifiers.clickbait final --config hand_tfidf_lr --in-domain dev`
- Huấn luyện lại trên toàn bộ dev, C=10 chọn bằng CV 5 phần trên dev.
- Ngưỡng 0,167, là ngưỡng tối ưu F1 trên điểm out-of-fold của CV 5 phần trên dev.
- PR-AUC vòng trong 0,573.

**Kết quả trên `corpus.csv` (1.244 bài):**
- 112 bài nhãn 1 (9,0%) theo cột `label`.
- 750 bài trong dev giữ nhãn sửa tay; model đồng ý với 99,1% số này vì đã học trên chúng.
- 494 bài ngoài dev: model gán 4,7% là clickbait. Phần này đã bị rút bớt mẫu dương khi chọn nhóm H, nên tỷ lệ thấp hơn mức 10% của phần dev ngẫu nhiên là hợp lý.

**Nếu PhoBERT hoặc Qwen cho kết quả tốt hơn rõ** (khoảng tin cậy không chồng nhau), chạy `final --from-file results/clickbait/kaggle/corpus_scores_<config>.csv`. Nếu không, giữ model hiện tại vì nó đơn giản hơn.

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
- **PhoBERT và Qwen:** chưa có kết quả. Kỳ vọng các model này xử lý được nhóm FN cần hiểu ngữ nghĩa. Chi phí GPU dự kiến khoảng 0,5 giờ cho PhoBERT và 2 giờ cho Qwen.

## 7. Chạy lại

```bash
conda create -n visim python=3.9 && conda activate visim
pip install -r requirements.txt pyvi
python -m src.classifiers.clickbait ladder          # mốc + cổ điển + ablation dữ liệu/đầu vào (~2 phút)
python -m src.classifiers.clickbait extras          # mốc thêm, ngưỡng, tách từ, cân bằng lớp, đường cong học (~1-2 phút)
python -m src.classifiers.clickbait in-domain       # model trong miền, CV lồng (~6 phút)
python -m src.classifiers.clickbait final --config hand_tfidf_lr --in-domain dev   # corpus_labels.csv
# Kaggle: python -m src.classifiers.clickbait kaggle-bundle, rồi làm theo notebooks/README_kaggle.md
python -m src.classifiers.clickbait collect         # gộp kết quả Kaggle vào results/clickbait/ladder_full.csv
# Chọn mẫu và áp nhãn cho một lượt sửa:
python -m src.classifiers.clickbait review --round N --config dev_scores_<config>.csv
python -m src.classifiers.clickbait apply-labels --round N   # đọc notes/clickbait_review/round_N.csv
```
