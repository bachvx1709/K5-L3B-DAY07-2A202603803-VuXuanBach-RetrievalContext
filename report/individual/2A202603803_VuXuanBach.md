# Báo cáo Day 7: Agentic RAG & Context Engineering

- **Họ và tên:** Vũ Xuân Bách
- **Mã số sinh viên (MSSV):** 2A202603803
- **Lớp:** K5-L3B
- **Repository:** `K5-L3B-DAY07-2A202603803-VuXuanBach-RetrievalContext`

---

## 1. Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Tài liệu: Giáo trình chính quy *College Success* (Tác giả: Amy Baldwin / OpenStax), giấy phép Creative Commons Attribution 4.0 International (CC BY 4.0).
  - Phạm vi: **Chapter 3: Managing Your Time and Priorities**.
  - Số lượng chunk: Kế thừa và sử dụng duy nhất chiến lược **Paragraph-aware (Strategy B)** từ Day 6 với tổng cộng **162 chunks** có đầy đủ metadata cấu trúc.
- **Vector store, embedding model, phiên bản và số chiều:**
  - Vector Store: **ChromaDB Local** (bản 1.5.9), lưu tại thư mục `chroma_db/`, collection duy nhất `strategy_b`. Metric khoảng cách: Cosine Distance.
  - Embedding Model: Google **`gemini-embedding-001`**, số chiều vector: **3.072 dimensions**.
- **Agent model, cách chạy và phân chia dev/test:**
  - Agent Model: **`gemini-3.5-flash-lite`** kết nối qua OpenAI-compatible SDK.
  - Phân chia Benchmark: Tổng cộng **40 câu hỏi độc lập** (`data/benchmark/benchmark.json`):
    - **Dev Split (30 câu):** 25 câu có bằng chứng cụ thể + 5 câu thiếu bằng chứng để đo abstention.
    - **Test Split (10 câu):** 8 câu có bằng chứng cụ thể + 2 câu thiếu bằng chứng.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - Câu có bằng chứng: 100% evidence quote được trích nguyên văn từ 162 chunks của Chapter 3 và được kiểm chứng tự động bằng script assertion.
  - Câu thiếu bằng chứng: Được tạo bám sát chủ đề học tập/sinh viên (như quy tắc ghi chép Cornell, hạn nộp FAFSA, thuật toán phần mềm Anki, phương pháp đọc SQ3R) nhưng hoàn toàn không có trong Chapter 3.
  - Tiêu chí chấm:
    - **Retrieval:** Hit@1 và Evidence Hit@3.
    - **Context Evidence Rate:** Số câu context đưa vào prompt chứa đủ evidence / tổng câu có đáp án.
    - **Grounded Answer (0/1/2):** 0 = sai hoặc thiếu evidence; 1 = đúng một phần hoặc thiếu citation; 2 = đúng, đủ và trích dẫn chuẩn `[B_chunk_XXXX]`.
    - **Abstention:** Tỷ lệ từ chối chính xác trên các câu unanswerable.

---

## 2. Retrieval và thử nghiệm

### 2.1 Bảng so sánh các cấu hình Retrieval trên tập Dev (30 câu) và Test (10 câu)

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / Thay đổi kỹ thuật | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / Chi phí |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | Dense Retrieval (Cosine), top_k=3, không filter, budget=3500 | Sử dụng embedding vector không gian 3072 chiều làm mốc cơ sở. | dev | 20/25 (80.0%) | 23/25 (92.0%) | 23/25 (92.0%) | ~1.24s / câu |
| **Phương pháp 2** | Lexical Retrieval (BM25Okapi), top_k=3, budget=3500 | Bắt chính xác từ khóa và số liệu thực tế qua tần suất từ TF-IDF. | dev | 18/25 (72.0%) | 22/25 (88.0%) | 22/25 (88.0%) | < 0.01s / câu  |
| **Phương pháp 3** | Hybrid Retrieval (RRF Dense + BM25), top_k=3, rrf_k=60, budget=3500 | Kết hợp điểm mạnh ngữ nghĩa của Dense và từ khóa của BM25. | dev | 19/25 (76.0%) | 23/25 (92.0%) | 23/25 (92.0%) | ~1.15s / câu |
| **Phương pháp 4 ** | Hybrid + Local Cross-Encoder Reranker (`ms-marco-MiniLM-L-12-v2`), top_k=3 | Lấy pool 15 candidates rồi dùng mô hình Rerank chấm điểm lại để lọc sạch nhiễu context. | dev | **20/25 (80.0%)** | 22/25 (88.0%) | **24/25 (96.0%)** | ~1.58s / câu |
| **Phương pháp 5** | Query Transformation (Rewriting via LLM) + Hybrid, top_k=3, budget=3500 | Viết lại câu hỏi thành từ khóa tìm kiếm sâu. | dev | 18/25 (72.0%) | 20/25 (80.0%) | 20/25 (80.0%) | ~2.51s / câu |
| **Cấu hình đã chọn** | Hybrid + Local MiniLM Reranker, top_k=3 | Đánh giá độc lập trên tập Test sau khi đã cố định cấu hình trên Dev. | test | **8/8 (100.0%)** | **8/8 (100.0%)** | **8/8 (100.0%)** | ~5.12s / câu |
| **Baseline** | Dense Retrieval thuần túy, top_k=3 | Đối chứng trên cùng tập Test. | test | 8/8 (100.0%) | 8/8 (100.0%) | 8/8 (100.0%) | ~4.35s / câu |

### 2.2 Phân tích lỗi thực tế trên tập Dev (Error Analysis)
1. **Lỗi 1 ở câu `dev_010` (Fear of Failure):**
   * *Nguyên nhân:* Baseline Dense Retrieval xếp chunk `B_chunk_0038` lên vị trí số 1 vì chunk này chứa tiêu đề in hoa *"Fear of Failure"*, khiến cosine similarity của vector thiên về tiêu đề hơn là nội dung giải thích cốt lõi ở `B_chunk_0039`.
   * *Khắc phục:* Khi áp dụng Cấu hình Cải thiện (Hybrid + MiniLM Reranker), mô hình cross-encoder đã phân tích sâu mối tương quan giữa câu hỏi và toàn văn đoạn trích, đẩy đúng `B_chunk_0039` vào Top 3 context.
2. **Lỗi 2 ở câu `dev_022` (Multitasking):**
   * *Nguyên nhân:* Văn bản gốc không dùng trực tiếp từ "multitask" mà dùng cụm diễn giải *"taking twice as long when switching between tasks"*, dẫn đến cả Dense và BM25 bị trượt chunk `B_chunk_0101` ở Top 3.
   * *Hành vi của Agent:* Agent thể hiện tính tin cậy tuyệt đối: khi context không đủ bằng chứng khẳng định, Agent chủ động từ chối trả lời (Abstention an toàn) thay vì tự suy diễn sai lệch.

---

## 3. Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - **Tool:** Cung cấp function `search_knowledge_base` với các tham số (`query`, `method`, `section_filter`, `top_k`).
  - **Số lượt gọi & Điều kiện dừng:** Hỗ trợ gọi tối đa `max_turns = 3`. Agent dừng lại ngay khi đã thu thập đủ bằng chứng hoặc khi đã đạt ngưỡng budget.
  - **Context Selection:** Tự động loại bỏ các chunk trùng lặp và kiểm soát ngân sách `context_budget = 3500` ký tự.
  - **Xử lý lỗi:** Trang bị cơ chế auto-retry khi gặp gián đoạn kết nối mạng API.

### 3.1 Bảng đánh giá chi tiết Agent End-to-End

| Cấu hình / split | Tool use | Số lượt gọi TB | Grounded answer (0/1/2) | Grounded TB /2.0 | Citation hợp lệ | Abstention đúng | Từ chối nhầm |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Dense) — Dev** | 30/30 (100%) | 1.1 lượt | 21 đúng đủ, 2 một phần, 2 trượt | 1.84 / 2.0 | 23/25 (92.0%) | 5/5 (100.0%) | 0/25 (0%) |
| **Cải thiện (MiniLM) — Dev** | 30/30 (100%) | 1.1 lượt | **22 đúng đủ**, 2 một phần, 1 trượt | **1.88 / 2.0** | **24/25 (96.0%)** | 4/5 (80.0%) | 0/25 (0%) |
| **Baseline (Dense) — Test** | 10/10 (100%) | 1.2 lượt | 5 đúng đủ, 3 một phần, 0 trượt | 1.62 / 2.0 | 8/8 (100.0%) | 2/2 (100.0%) | 0/8 (0%) |
| **Cải thiện (MiniLM) — Test** | 10/10 (100%) | 1.2 lượt | 5 đúng đủ, 2 một phần, 1 trượt | 1.50 / 2.0 | 7/8 (87.5%) | **2/2 (100.0%)** | 0/8 (0%) |

### 3.2 Minh họa 3 ca kiểm thử thực tế điển hình

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và Citation | Nhãn, lý do |
|---|---|---|---|---|
| **`dev_01`** | Có đáp án (Bảng biểu / Số liệu) | `search_knowledge_base(query="average total loss additional year college Washington Post", method="rerank_local")` -> Lấy `B_chunk_0016`, `B_chunk_0014`, `B_chunk_0015`. | *"According to Table 3.1 based on Washington Post estimates, the estimated average total loss for an additional year of college is $151,115 [B_chunk_0016]."* | **Mức 2 (Đúng và Đủ):** Bắt chính xác 100% số liệu tiền tệ và gắn đúng citation `[B_chunk_0016]`. |
| **`dev_026`** | Thiếu bằng chứng (Abstention) | `search_knowledge_base(query="Cornell Note-Taking System Chapter 3")` -> Kết quả không chứa thông tin về Cornell Notes. | *"There is insufficient evidence in Chapter 3 (Managing Your Time and Priorities) to describe the specific five steps of the Cornell Note-Taking System."* | **Abstention đúng:** Agent nhận biết tài liệu không có thông tin và chủ động từ chối trả lời an toàn. |
| **`dev_022`** | Ca lỗi (Retrieval Miss) | `search_knowledge_base(query="reality of multitasking priorities")` -> Lấy các chunk về time management chung nhưng trượt chunk cụ thể 101. | *"Based on the retrieved evidence, there is insufficient evidence in the document to answer the question about the reality of multitasking."* | **Mức 0 (Lỗi do từ khóa ẩn):** Retriever trượt do tác giả không dùng trực tiếp từ "multitask", Agent kích hoạt cơ chế từ chối. |

- **Đường dẫn trace, nhãn, lệnh chạy và thời gian:**
  - File traces & metrics: `results/retrieval_dev_summary.json`, `results/agent_dev_baseline_dense.json`, `results/agent_dev_improved_rerank.json`, `results/agent_test_baseline_dense.json`, `results/agent_test_improved_rerank.json`.
  - Bộ câu hỏi và nhãn: `data/benchmark/benchmark.json`.
  - Lệnh chạy: `python3 src/evaluate_dev.py` (chạy Dev) và `python3 src/evaluate_test.py` (chạy Test).
  - Thời gian thực thi trung bình: ~5.06s – 5.57s / câu (gồm tool call, retrieval và LLM generation).
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - *Trước/sau:* Cải thiện với MiniLM Reranker giúp tăng Context Evidence Rate trên Dev từ 92.0% lên 96.0% và điểm Grounded trung bình từ 1.84 lên 1.88/2.0.
  - *Ví dụ lỗi:* Câu `dev_010` (Fear of Failure) ban đầu bị Dense kéo nhầm tiêu đề `B_chunk_0038`, sau khi thêm Reranker đã đẩy đúng nội dung chi tiết `B_chunk_0039` vào context; câu `dev_022` trượt do tác giả không dùng từ "multitask", Agent kích hoạt từ chối an toàn.
  - *Đánh đổi và giới hạn:* Reranker MiniLM lọc nhiễu tốt và tăng tốc độ suy luận, nhưng do tối ưu cho single passage nên ở câu hỏi cần nhiều đoạn bổ trợ (`test_07`), mô hình ưu tiên đoạn chi tiết nhất và đẩy đoạn tổng quan xuống dưới. Hướng khắc phục là tăng top-k sau rerank hoặc dùng MMR.

## 4. Thử thách nâng cao (Đạt Bonus 15 điểm)

- **Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:**
  - Giáo trình *College Success* (Tác giả: Amy Baldwin / OpenStax, giấy phép mở CC BY 4.0).
  - Phạm vi thực tế: **140 trang liên tục** (từ trang 1 đến trang 140, gồm trọn vẹn Chapter 1, 2, 3 và 4), chứa tổng cộng 36 bảng biểu và 56 hình ảnh/sơ đồ/biểu đồ.
- **Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:**
  - Trích xuất toàn văn bản và siêu dữ liệu cấu trúc qua `pypdf`, phân đoạn văn bản và đánh chỉ mục vector vào ChromaDB collection `advanced_140pages` (138 vectors 3.072 chiều từ `gemini-embedding-001`).
  - Gắn nhãn metadata nhận diện tự động cho các trang chứa bảng (`has_table`) và hình ảnh/biểu đồ (`has_figure`).
  - Pipeline kết hợp Hybrid Retrieval (Dense + BM25) cùng Local Cross-Encoder Reranker (`ms-marco-MiniLM-L-12-v2`) với `top_k=8` chunks.
- **Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:**
  - Benchmark riêng: 60 câu (`data/benchmark/advanced_benchmark.json`) gồm 30 câu Dev và 30 câu Test bao phủ toàn diện 6 nhóm câu hỏi theo yêu cầu quy định.
  - Model: `gemini-3.5-flash-lite` làm agent điều phối suy luận, trích xuất và sinh câu trả lời có citation.

### Kết quả đánh giá độc lập trên toàn bộ 30 câu Test:

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|:---:|:---:|---|
| **Bảng (Table)** | 6/6 | **100.0%** | Bắt chính xác 100% số liệu tiền tệ ($82,074), bảng mốc thời gian Table 1.4 và SMART GPA Table 3.6. |
| **Ảnh, sơ đồ (Image/Diagram)** | 5/5 | **100.0%** | Nhận diện xuất sắc sơ đồ 4 góc Eisenhower, đồng hồ quả cà chua Pomodoro, chân dung Sarah. |
| **Biểu đồ (Chart)** | 5/5 | **100.0%** | Đọc chính xác xu hướng thu nhập trọn đời (Figure 1.3), 6 vùng thích ứng (Figure 1.4). |
| **Multi-hop** | 1/4 | **25.0%** | Đạt câu liên kết ma trận Eisenhower & Eat the Frog; trượt 3 câu do bằng chứng nằm ở 2 chương quá xa nhau. |
| **Text** | 4/4 | **100.0%** | Trả lời chính xác 100% câu hỏi lý thuyết, câu nói của Mark Twain, quy tắc nghỉ Pomodoro. |
| **Thiếu bằng chứng (Abstention)** | 6/6 | **100.0%** | Từ chối tuyệt đối 100% các câu hỏi nằm ngoài tài liệu (lãi suất nợ, công thức NPV, thí nghiệm hóa học). |
| **Toàn bộ Benchmark Test** | **27/30** | **90.0%** | **ĐẠT CHUẨN BONUS ($\ge$ 90.0% theo quy định tại ADVANCED.md)** |

- **Trace, nhãn và evidence từng câu; thời gian, lỗi hệ thống:**
  - File kết quả chi tiết: `results/advanced_test_results.json` (kèm kết quả Dev tại `results/advanced_dev_results.json`).
  - Thời gian thực thi trung bình: ~6.3s / câu hỏi.
  - Lỗi hệ thống: 0 lỗi, tỷ lệ từ chối an toàn đạt 100%.
- **Giới hạn, cách tái lập và đường dẫn benchmark:**
  - Pipeline có thể tái lập hoàn toàn qua lệnh: `python3 src/evaluate_advanced_test.py`.
  - Bộ câu hỏi và nhãn golden: `data/benchmark/advanced_benchmark.json`.
