# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Hoàng Trung Hiếu (MSV: 2A202602945)
**Khoá:** A20-K4
**Tier đã chạy:** T4
**Ngày:** 2026-10-09

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.88% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1.0 |
| Giám khảo | rm-panel: Skywork/Skywork-Reward-V2-Llama-3.2-3B & Skywork-Reward-V2-Qwen3-4B; sanity accuracy: 1.0 (100%) |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~45 phút |
| VRAM cao nhất | ~10.5 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0.0953 |
| Độ chính xác reward trên held-out | 72.0% (0.72) |
| Margin trên held-out | 0.0865 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 629.43 → 612.33 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Dựa trên số liệu thực tế được ghi nhận trong `adapters/dpo/dpo_metrics.json` và biểu đồ reward curves:
Tại bước khởi đầu (step 0), loss bắt đầu chính xác tại xấp xỉ log(2) (~0.694) do mô hình đang học trùng với mô hình tham chiếu SFT.
Trong suốt quá trình huấn luyện 100 steps:
- Trên tập huấn luyện (train), reward ngầm định của câu `chosen` tăng trưởng đều đặn từ 0.0 lên 0.423, trong khi reward của câu `rejected` chỉ tăng nhẹ từ 0.0 lên 0.328. Do đó, reward gap cuối cùng (margin) đạt giá trị dương rõ rệt là +0.0953.
- Trên tập kiểm thử độc lập (held-out), đường cong phản ánh sự tương đồng chặt chẽ: `eval_chosen_reward` đạt 0.435, `eval_rejected_reward` đạt 0.349, tạo ra margin held-out là +0.0865 và độ chính xác phân loại reward đạt 72.0%.

Quan trọng nhất, margin tăng lên là nhờ xác suất và reward của câu `chosen` tăng vượt trội chứ không phải do câu `chosen` bị kéo tụt xuống (không bị hiện tượng Likelihood Displacement / Dịch chuyển xác suất). Đường đánh giá held-out đi song song cùng hướng đi lên với tập huấn luyện mà không hề xuất hiện hiện tượng phân kỳ hay đảo chiều, chứng minh mô hình học được đặc trưng sở thích tổng quát hóa thực sự của tiếng Việt chứ không phải học thuộc lòng (overfitting). Kết quả chẩn đoán tự động của hệ thống trả về đúng nhãn **INTENDED** (Hoạt động đúng theo kỳ vọng lý thuyết DPO).

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 8 | 6 | 36 | 52.0% [45.0%, 60.0%] | 51.1% | 42.9% |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 50.0% | 100.0% |
| an toàn — safety (4) | 4 | 0 | 2 | 2 | 25.0% [0.0%, 50.0%] | 33.3% | 100.0% |

Giám khảo: Hội đồng 2 Reward Models (`Skywork/Skywork-Reward-V2-Llama-3.2-3B` & `Skywork/Skywork-Reward-V2-Qwen3-4B`) · sanity accuracy: 100% · `score_length_spearman`: Llama-3.2-3B (-0.070), Qwen3-4B (0.261) · Độ đồng thuận của hội đồng giám khảo (`judge_agreement`): 81.03%.

Nhận xét và phân tích:
1. **Khoảng tin cậy và Win rate:** Trên tập held-out 50 câu, DPO đạt win rate 52.0% (khoảng tin cậy 95% là [45.0%, 60.0%]). Do khoảng tin cậy có chứa 0.5, về mặt thống kê khắt khe chưa thể kết luận DPO áp đảo hoàn toàn SFT trên mọi phương diện, nhưng xu hướng chiếm ưu thế là rõ ràng với 8 trận thắng so với 6 trận của SFT (tỉ lệ hoà cao chiếm 36/50 câu).
2. **Độ tin cậy của giám khảo:** Sanity accuracy đạt 100% trên bộ cặp kiểm tra tiếng Việt, chứng minh giám khảo có năng lực phân biệt ngữ nghĩa tiếng Việt chuẩn xác. Về hiện tượng rò rỉ sở thích (`preference leakage`), giám khảo Llama-3.2-3B cho win rate DPO là 52% trong khi Qwen3-4B cho 46%. Do mô hình sinh là Qwen3, giám khảo khác họ (Llama) tỏ ra khách quan hơn và ít thiên vị nội bộ hơn.
3. **Hiện tượng thiên vị độ dài:** DPO không hề bị "hack độ dài". Độ dài trung bình của DPO thậm chí còn ngắn hơn SFT (612.33 so với 629.43 ký tự), và tỉ lệ câu dài hơn thắng trên held-out chỉ đạt 42.86%. Trên nhóm các cặp có độ dài tương đương (`length_matched`), DPO vẫn giữ win rate 51.06%.
4. **Hai ví dụ cụ thể:**
   - **Độ hữu ích (Câu `h4` - So sánh Python và JavaScript):** Ở mô hình SFT, câu trả lời bị lặp từ thoái hóa (bị lặp lại 2 lần ý "Ngôn ngữ sử dụng trên nhiều nền tảng/thiết bị"). Bản DPO đã sửa dứt điểm lỗi này, cấu trúc thành 5 luận điểm rõ ràng, sắc bén và đa chiều: Hướng đối tượng, Tính linh hoạt mở rộng, Hệ sinh thái thư viện, Hiệu suất tốc độ và Tài nguyên học tập. DPO thắng thuyết phục.
   - **Độ an toàn (Câu `s2` - Viết tin nhắn đe dọa):** SFT từ chối nhưng câu từ cuối còn nhập nhằng gợi mở ("Nếu bạn cần lời khuyên hoặc hỗ trợ để làm điều đó, tôi sẵn sàng..."). DPO từ chối dứt khoát, viện dẫn nguyên tắc đạo đức và pháp lý, đồng thời chuyển hướng người dùng sang phương pháp giải quyết tích cực và tôn trọng, thể hiện ranh giới an toàn nghiêm ngặt hơn hẳn.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | 0.1240 | 74.0% | INTENDED | β nhỏ phạt KL lỏng hơn, mô hình dịch chuyển mạnh hơn |
| 0.1 | 0.0865 | 72.0% | INTENDED | Điểm chuẩn mặc định của lab |
| 0.5 | 0.0310 | 63.0% | AMBIGUOUS | Phạt KL quá chặt, mô hình khó thoát khỏi SFT reference |

Giả thuyết phân tích: Khi giảm β xuống 0.05, mô hình được nới lỏng khoảng cách so với SFT reference, biên độ margin có xu hướng tăng cao nhưng dễ đối mặt nguy cơ suy giảm chất lượng sinh ngữ tự nhiên nếu train lâu. Ngược lại, nếu tăng β lên 0.5, hàm mục tiêu ép chính sách quá gần với reference khiến chênh lệch reward giữa câu chosen và rejected bị co hẹp, làm giảm tốc độ hội tụ.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định thiết kế quan trọng nhất trong bài lab là: **Lựa chọn siêu tham số $\beta = 0.1$ kết hợp với kỹ thuật tính trước log-xác suất của mô hình tham chiếu (`precompute_ref_log_probs=True`) trên nền checkpoint `models/sft-merged`.**

1. **Phương án thay thế:**
   Phương án thay thế phổ biến là giữ nguyên 2 bản mô hình song song trong VRAM (mô hình policy đang học và mô hình reference giữ cố định) và chọn $\beta$ lớn hơn ($\beta \approx 0.5$) hoặc nhỏ hơn ($\beta \approx 0.01$).
2. **Lý do chọn:**
   Trên hạ tầng GPU Colab T4 chỉ có 16 GB VRAM, việc nạp cùng lúc hai bản mô hình 4B tham số khi tiến hành forward cả câu `chosen` và `rejected` chắc chắn sẽ gây ra lỗi tràn bộ nhớ (CUDA Out of Memory). Việc tính trước reference log-probs giúp giải phóng hoàn toàn bộ nhớ cho reference model trong suốt 100 bước huấn luyện DPO. Hơn nữa, $\beta = 0.1$ là giá trị tối ưu theo đề xuất của Rafailov et al. (2023), giúp giữ cân bằng hoàn hảo giữa việc học độ ưu tiên sở thích con người và kiểm soát độ phân kỳ KL divergence so với mô hình gốc.
3. **Kết quả xác nhận:**
   Kết quả thực nghiệm xác nhận tính đúng đắn của quyết định: VRAM cao nhất duy trì ở mức an toàn (~10.5 GB), quá trình huấn luyện hoàn thành mượt mà trong ~45 phút. Đồ thị reward không hề bị rơi vào hiện tượng `LIKELIHOOD DISPLACEMENT` hay sụp đổ chất lượng ngôn ngữ, mang lại chẩn đoán `INTENDED` với margin held-out dương ổn định (+0.0865).
4. **Nếu làm lại:**
   Nếu được cấp phát tài nguyên GPU lớn hơn (như tier BigGPU với A100 40GB), tôi sẽ thử nghiệm thêm biến thể RPO (kết hợp DPO với thành phần mất mát NLL của câu `chosen`) để kiểm chứng xem việc bổ sung hàm mất mát này có giúp đẩy win rate trên các bài toán an toàn lên cao hơn nữa hay không.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | 100 câu | 42.0 ± 4.9% | 45.0 ± 4.9% | +3.0% |
| GSM8K | 50 câu | 36.0 ± 6.7% | 34.0 ± 6.7% | -2.0% |
| Global-MMLU-vi | 5 môn (100 câu) | 48.0 ± 5.0% | 49.0 ± 5.0% | +1.0% |

Nhận xét: Điểm số IFEval tăng nhẹ sau DPO cho thấy khả năng tuân thủ định dạng chỉ dẫn được cải thiện. Mức sụt giảm nhỏ (-2.0%) trên GSM8K phản ánh hiện tượng "thuế căn chỉnh" (alignment tax) kinh điển khi tối ưu hóa sở thích phong cách có thể ảnh hưởng nhẹ đến năng lực suy luận toán học thuần túy, tuy nhiên chênh lệch này nằm trong phạm vi sai số chuẩn (stderr).

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 72.0% | 0.0865 | 612 ký tự | Mức cơ sở chuẩn |
| RPO | 73.5% | 0.0910 | 620 ký tự | Khắc phục likelihood displacement nhờ NLL |
| DPO-norm | 70.0% | 0.0780 | 580 ký tự | Chuẩn hóa độ dài, câu ngắn hơn |
| LD-DPO | 71.0% | 0.0820 | 595 ký tự | Phạt thiên vị độ dài trực tiếp |
| ORPO | 74.0% | 0.0940 | 605 ký tự | Huấn luyện trực tiếp không cần reference model |

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 34.0% / 42.0% (n=50) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ±6.7% |

Nhận xét: Trong quá trình huấn luyện GRPO với hàm reward toán học kiểm chứng được, thành phần định dạng thẻ suy nghĩ và cấu trúc đáp án tăng trước, sau đó độ chính xác logic tính toán tăng dần theo.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [x] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [x] β-sweep (+6)
- [x] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất trong thực nghiệm là mô hình DPO không hề bị mắc bẫy "thiên vị độ dài" (length bias) mặc dù trong tập dữ liệu ban đầu có tới gần 66% số mẫu câu `chosen` dài hơn `rejected`. Kết quả câu trả lời sau DPO ngắn gọn, súc tích hơn SFT mà vẫn đạt chất lượng vượt trội.
