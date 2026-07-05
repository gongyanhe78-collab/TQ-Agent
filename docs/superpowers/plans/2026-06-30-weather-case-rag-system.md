# Weather Case RAG System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end pipeline that extracts weather disaster cases from monthly PDF reports, saves one case per TXT file, indexes the cases in ChromaDB, and exposes a Vue3 + Element Plus RAG interface that can hit relevant cases for the questions in `weather_qa_results.json`.

**Architecture:** Use a Python backend to run the extraction, indexing, retrieval, reranking, and RAG flow. Use a Vue3 frontend for browsing case IDs and chatting against the indexed case library. Keep every intermediate artifact on disk for auditability.

**Tech Stack:** Python 3.10, FastAPI, ChromaDB, OpenAI-compatible DashScope APIs, Vue3, Element Plus, Vite

---

### Task 1: Project Skeleton

**Files:**
- Create: `requirements.txt`
- Create: `backend/app/__init__.py`
- Create: `frontend/package.json`
- Create: `frontend/vite.config.js`
- Create: `README.md`

- [ ] **Step 1: Add backend and frontend dependency manifests**
- [ ] **Step 2: Add backend/frontend directory skeleton**
- [ ] **Step 3: Add local run instructions**

### Task 2: Extraction Tests First

**Files:**
- Test: `tests/test_case_splitter.py`
- Test: `tests/test_extraction_pipeline.py`

- [ ] **Step 1: Write failing tests for section splitting and case TXT formatting**
- [ ] **Step 2: Run `python -m unittest discover -s tests -v` and confirm failures**

### Task 3: Extraction Implementation

**Files:**
- Create: `backend/app/config.py`
- Create: `backend/app/models.py`
- Create: `backend/app/services/text_cleaning.py`
- Create: `backend/app/services/case_splitter.py`
- Create: `backend/app/services/pdf_reader.py`
- Create: `backend/app/services/llm_client.py`
- Create: `backend/app/services/extraction_pipeline.py`

- [ ] **Step 1: Implement PDF text loading, rule-first splitting, and LLM refinement hooks**
- [ ] **Step 2: Implement TXT persistence with `source_pdf`, `case_id`, `title`, `date_range`, and `content`**
- [ ] **Step 3: Re-run extraction tests and confirm green**

### Task 4: Retrieval Tests First

**Files:**
- Test: `tests/test_vector_store.py`
- Test: `tests/test_rag_service.py`

- [ ] **Step 1: Write failing tests for vector round-trip and reranked retrieval**
- [ ] **Step 2: Run `python -m unittest discover -s tests -v` and confirm failures**

### Task 5: Retrieval Implementation

**Files:**
- Create: `backend/app/services/embedding_client.py`
- Create: `backend/app/services/vector_store.py`
- Create: `backend/app/services/retrieval.py`

- [ ] **Step 1: Implement embedding client and Chroma-backed case store**
- [ ] **Step 2: Implement retrieval and reranking pipeline**
- [ ] **Step 3: Re-run tests and confirm green**

### Task 6: API and Frontend

**Files:**
- Create: `backend/app/main.py`
- Create: `frontend/index.html`
- Create: `frontend/src/main.js`
- Create: `frontend/src/App.vue`
- Create: `frontend/src/api.js`
- Create: `frontend/src/components/CaseList.vue`
- Create: `frontend/src/components/ChatPanel.vue`

- [ ] **Step 1: Expose extraction, indexing, listing, detail, and RAG query endpoints**
- [ ] **Step 2: Build a simple Vue3 + Element Plus UI**
- [ ] **Step 3: Verify the UI can render indexed case IDs and answer queries through the backend**

### Task 7: Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run backend unit tests**
- [ ] **Step 2: Install dependencies if approved and run API/frontend verification commands**
- [ ] **Step 3: Document any remaining setup for live PDF extraction and DashScope calls**
