# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Bình Thành
**Nhóm:** 29
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Trong không gian vector, vector A "gần" vector B hơn, cụ thể thì nếu hai vector có hướng gần nhau sẽ làm góc giữa chúng nhỏ hơn -> gần hơn trong không gian vector 

**Ví dụ HIGH similarity:**
- Sentence A: con mèo
- Sentence B: mèo con
- Tại sao tương đồng: đều thuộc vùng "mèo" trong vector space

**Ví dụ LOW similarity:**
- Sentence A: con mèo
- Sentence B: con chó
- Tại sao khác: cùng là động vật nhưng khác thực thể, nếu không chung 1 ngữ cảnh hai sentence sẽ chiếm vị trí khác nhau trong không gian vector

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity tập trung vào hướng của vector thay vì độ lớn, còn Euclidean distance tập trung vào khoảng cách. Khi so sánh ngữ nghĩa, độ dài vector không phản ánh trực tiếp mức giống nghĩa, vì vậy cosine thường phù hợp hơn Euclidean distance.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* dùng công thức num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap)). đơn giản thôi ceil(9950 / 450) = ceil(22.111) = 23
> *Đáp án:* 23

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Overlap tăng làm step giảm, nên số chunk tăng. Overlap nhiều thì sẽ làm tăng số chunk, em chưa hiểu câu hỏi? tại sao lại mặc định là muốn overlap? overlap có thể có lợi và hại mà 

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Finance

**Tại sao nhóm chọn domain này?**
> Tại vì nó khó, tài liệu tài chính có nhiều số liệu, cần độ chính xác.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | tsla-20251231.pdf (Tesla 10-K Annual Report 2025, 169 trang) | SEC EDGAR / Tesla IR | 395,205 | source, company, doc_type, fiscal_year, chunk_index |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | string | `"tsla-20251231.pdf"` | Xác định tài liệu gốc, hỗ trợ filter theo file và delete_document |
| `company` | string | `"Tesla"` | Filter kết quả theo công ty khi có nhiều tài liệu nhiều công ty |
| `doc_type` | string | `"10-K Annual Report"` | Phân loại loại báo cáo (10-K, 10-Q, earnings call...) |
| `fiscal_year` | int | `2025` | Filter theo năm tài chính, tránh trộn số liệu nhiều năm |
| `chunk_index` | int | `42` | Xác định vị trí chunk trong tài liệu, hỗ trợ lấy context xung quanh |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| tsla-20251231.pdf (395,205 ký tự) | FixedSizeChunker (`fixed_size`) | 1098 | ~360 ký tự | Trung bình — cắt cứng, không theo ranh giới câu |
| tsla-20251231.pdf (395,205 ký tự) | SentenceChunker (`by_sentences`) | 642 | ~615 ký tự | Tốt — giữ nguyên câu hoàn chỉnh |
| tsla-20251231.pdf (395,205 ký tự) | RecursiveChunker (`recursive`) | 1206 | ~328 ký tự | Tốt — ưu tiên tách theo đoạn/câu trước khi cắt |
| tsla-20251231.pdf (395,205 ký tự) | ParagraphChunker (`paragraph`) | 615 | ~800 ký tự | Tốt — giữ nguyên đoạn văn, packing greedy theo `max_chunk_size`; bị ảnh hưởng bởi PDF header noise |

### Strategy Của Tôi

**Loại:** FixedSizeChunker

**Mô tả cách hoạt động:**
> FixedSizeChunker chia văn bản thành các đoạn có độ dài cố định (tính theo số ký tự), với một khoảng overlap giữa các chunk liền kề để tránh mất thông tin tại điểm cắt. Cách chia không dựa vào bất kỳ dấu hiệu ngôn ngữ nào (câu, đoạn văn), mà đơn thuần trượt cửa sổ kích thước cố định qua toàn bộ văn bản. Kết quả là số chunk và độ dài chunk hoàn toàn có thể dự đoán được từ tham số đầu vào.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu tài chính như 10-K có nhiều bảng số liệu và danh sách bullet point — các cấu trúc này không nhất thiết phải giữ nguyên câu để vẫn mang ý nghĩa. FixedSizeChunker đảm bảo mỗi embedding có kích thước nhất quán, tránh hiện tượng embedding lệch do độ dài quá chênh lệch. Ngoài ra tốc độ chunking nhanh phù hợp khi cần xử lý tài liệu lớn như báo cáo thường niên 395K ký tự.

**Code snippet (nếu custom):**
> Sử dụng built-in `FixedSizeChunker`, không custom thêm.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| tsla-20251231.pdf | SentenceChunker (best baseline) | 642 | ~615 ký tự | 6.65/10 avg |
| tsla-20251231.pdf | **FixedSizeChunker (của tôi)** | 1098 | ~360 ký tự | 5.96/10 avg |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Nguyễn Bình Thành (tôi) | FixedSizeChunker | 5.96 | Đơn giản, nhất quán, chunk count có thể kiểm soát | Cắt giữa câu, mất ngữ cảnh tại biên chunk |
| Hàn Quang Hiếu | RecursiveChunker | 6.02 | Linh hoạt, ưu tiên tách theo đoạn/câu | Chunk count cao nhất (1206), nhiều chunk nhỏ |
| Phan Anh Khôi | SentenceChunker | 6.65 | Giữ câu hoàn chỉnh, embedding ngữ nghĩa chính xác hơn | Chunk count thấp nhất, câu dài vẫn có thể mang nhiều ý |
| Trương Quang Lộc | ParagraphChunker | 6.00 | Giữ nguyên đoạn văn, chunk mang ý nghĩa hoàn chỉnh hơn | PDF header "Tesla, Inc." tạo noise chunk — Q1 top-1 lệch, Q2 hallucinate HQ, Q3 miss hoàn toàn |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> SentenceChunker cho retrieval score cao nhất (6.65/10) trên domain tài chính vì tài liệu 10-K có nhiều câu hoàn chỉnh chứa số liệu quan trọng — giữ nguyên câu giúp embedding capture đúng ngữ nghĩa hơn. FixedSizeChunker nhanh và đơn giản nhưng dễ cắt đứt giữa câu, đặc biệt bất lợi với văn bản dày đặc số liệu như báo cáo tài chính.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Tách câu bằng regex theo dấu kết câu (., !, ?) kết hợp khoảng trắng hoặc xuống dòng: `(?<=[.!?])(?:\s+|\n+)`. Sau khi split sẽ strip và loại phần rỗng để tránh chunk rác. Cuối cùng gom câu theo `max_sentences_per_chunk` để đảm bảo output ổn định.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán thử tách theo thứ tự separator ưu tiên (`\n\n`, `\n`, `. `, ` `, rồi fallback ký tự). Base case: chuỗi rỗng thì bỏ qua, chuỗi có độ dài <= `chunk_size` thì trả về ngay, hết separator thì cắt cứng theo `chunk_size`. Trong quá trình split có bước "pack" lại các phần nhỏ để tạo chunk dài nhất có thể mà vẫn không vượt ngưỡng.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi document được chuẩn hóa thành record gồm `id`, `content`, `metadata`, `embedding`; metadata luôn gắn thêm `doc_id` để hỗ trợ delete theo tài liệu gốc. `add_documents` tạo embedding một lần và lưu vào store in-memory; nếu Chroma khả dụng thì đồng bộ thêm vào collection. `search` embed query rồi chấm điểm bằng dot product với từng embedding, sau đó sort giảm dần theo score.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` thực hiện filter metadata trước, rồi mới chạy similarity trên tập con để tăng precision. Điều kiện filter dùng match đầy đủ theo cặp key-value trong `metadata_filter`. `delete_document` xóa toàn bộ record có `metadata.doc_id == doc_id`, đồng thời thử xóa trên Chroma nếu backend này đang bật.

### KnowledgeBaseAgent

**`answer`** — approach:
> Agent retrieve top-k chunks từ store, rồi ghép thành context có kèm score để tăng tính minh bạch. Prompt được cấu trúc theo dạng instruction + context + question, yêu cầu model chỉ trả lời dựa trên context và nói rõ khi thiếu dữ kiện. Cách này giúp hạn chế hallucination và bám sát RAG pattern.

### Test Results

```
# Paste output of: pytest tests/ -v
===================================== 42 passed in 1.05s ======================================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Tesla had 134,785 employees as of December 31, 2025 | Tesla's global workforce numbered 134,785 people at year-end | high | 0.95 | ✅ |
| 2 | Tesla pays no cash dividends on its common stock | Tesla does not distribute dividends to shareholders | high | 0.91 | ✅ |
| 3 | Cybercab is Tesla's purpose-built robotaxi product | Tesla Autopilot is a driver-assistance software feature | low | 0.61 | ✅ |
| 4 | Tesla total revenues for fiscal year 2025 | Tesla annual sales and financial performance figures | high | 0.84 | ✅ |
| 5 | Gigafactory Texas is located in Austin | Tesla's solar panel installation process for residential homes | low | 0.38 | ✅ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 3 bất ngờ nhất — "Cybercab robotaxi" và "Autopilot software feature" đều là sản phẩm Tesla liên quan đến autonomous driving nhưng score chỉ 0.61, thấp hơn dự kiến. Điều này cho thấy embedding phân biệt được mức độ trừu tượng: một bên là phần cứng (xe), một bên là phần mềm (tính năng), dù cùng ngữ cảnh công ty. Embeddings không chỉ nắm bắt từ khóa chung mà còn encode loại thực thể và vai trò của nó trong câu.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | How many employees did Tesla have as of end of 2025? | As of December 31, 2025, Tesla had 134,785 employees worldwide. |
| 2 | Where is Tesla headquartered and what are its primary manufacturing locations? | Tesla is headquartered in Austin, Texas. Primary owned manufacturing facilities include Gigafactory Texas (Austin), Fremont Factory (California), Gigafactory Nevada (Sparks), and Gigafactory Berlin-Brandenburg (Germany). Gigafactory Shanghai and Megafactory Shanghai are owned buildings on leased land. |
| 3 | Does Tesla pay dividends to its shareholders? | Tesla has never declared or paid cash dividends on its common stock and does not anticipate paying any in the foreseeable future. |
| 4 | What autonomous vehicle product is Tesla developing for the robotaxi market? | Tesla is developing Cybercab, a purpose-built Robotaxi product, alongside its FSD (Supervised) and neural network capabilities to compete in the autonomous vehicle and ride-hailing market. |
| 5 | How does Tesla protect its intellectual property while still supporting EV industry growth? | Tesla seeks patent protection broadly but has pledged not to initiate lawsuits against parties that infringe its patents through activity relating to electric vehicles or related equipment, as long as they act in good faith — to encourage development of a common EV platform. |

### Kết Quả Của Tôi

> Sử dụng **FixedSizeChunker**.

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) | Điểm |
|---|-------|--------------------------------|-------|-----------|------------------------|------|
| 1 | Tesla employees end of 2025? | "29,000 employees globally advancing their careers in 2025..." (chunk về career programs, không phải headcount) | 0.6139 | Partial | Answer đúng (134,785) nhờ chunk top-2 có số liệu; relevant chunk không ở top-1. | 1/2 |
| 2 | Tesla HQ & primary manufacturing locations? | Bảng primary manufacturing facilities (Gigafactory Texas, Fremont, Gigafactory NV, Berlin-Brandenburg, Gigafactory Shanghai, Megafactory Shanghai, Gigafactory NY, Megafactory Lathrop) | 0.5693 | Yes | HQ: Austin, TX. Danh sách đầy đủ các cơ sở owned và leased. | 2/2 |
| 3 | Tesla pay dividends? | Consolidated Statements of Equity — không liên quan đến dividend policy | 0.5665 | No | Top-3 không có chunk nào về chính sách cổ tức. AI answer: "does not specify." | 0/2 |
| 4 | Tesla autonomous vehicle for robotaxi market? | "Our Robotaxi business currently operates with Model Y vehicles... products such as FSD (Supervised)" | 0.6372 | Yes | Tesla is developing the Robotaxi product, specifically the Cybercab. | 2/2 |
| 5 | Tesla IP protection & EV growth? | "irrevocably pledged that we will not initiate a lawsuit against any party for infringing our patents through activity relating to electric vehicles..." | 0.5907 | Yes | Tesla bảo vệ IP qua patent, cam kết không kiện bên vi phạm bằng sáng chế EV miễn họ hành động thiện chí. | 2/2 |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5 (Q3 miss hoàn toàn)

**Tổng điểm Results:** 7 / 10

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Học cách Hàn Quang Hiếu lấy BCTC

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Không có gì ấn tượng, quan sát thấy phần lớn các bạn toàn để AI hallucinate ra benchmark của riêng mình, tự đo chứ không theo SCORING.md gì

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Chắc sẽ đọc kĩ bài toán để hiểu techinique chunking hơn, thay vì để AI handle phần lớn đoạn này

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | 5 | / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **84 / 100** |
