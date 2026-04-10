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
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| | | | |
| | | | |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| | FixedSizeChunker (`fixed_size`) | | | |
| | SentenceChunker (`by_sentences`) | | | |
| | RecursiveChunker (`recursive`) | | | |

### Strategy Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / custom strategy]

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline | | | |
| | **của tôi** | | | |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

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
| 1 | | | high / low | | |
| 2 | | | high / low | | |
| 3 | | | high / low | | |
| 4 | | | high / low | | |
| 5 | | | high / low | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
