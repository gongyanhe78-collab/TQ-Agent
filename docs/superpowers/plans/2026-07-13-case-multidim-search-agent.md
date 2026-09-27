# Case Multidimensional Search Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portable LangChain-based hybrid case-search agent with Postman-testable APIs, deterministic charting, CSV/XLSX export, and a Chinese PDF report containing relevant charts and evidence images.

**Architecture:** All production code lives in `backend/app/services/agent/case_multidim_search/`. The package receives existing stores through dependency injection, applies hard structured filters, maps vector-retrieved chunks to standard cases, extracts traceable intensity metrics, aggregates results, generates charts, and renders a PDF. `backend/app/main.py` only registers the package router and configures dependencies.

**Tech Stack:** Python 3.10, FastAPI, Pydantic, LangChain, ChromaDB, Matplotlib, Pillow, pypdfium2, unittest.

---

### Task 1: Schemas and Structured Retrieval

**Files:**
- Create: `backend/app/services/agent/case_multidim_search/__init__.py`
- Create: `backend/app/services/agent/case_multidim_search/schemas.py`
- Create: `backend/app/services/agent/case_multidim_search/query_parser.py`
- Create: `backend/app/services/agent/case_multidim_search/structured_retriever.py`
- Test: `tests/case_multidim_search/test_structured_retriever.py`

- [ ] Write tests proving combined date, disaster, city, image-type and data-category filters are hard constraints.
- [ ] Run `python -m unittest tests.case_multidim_search.test_structured_retriever -v` and verify failures are caused by missing modules.
- [ ] Define Pydantic request and response models plus a deterministic Chinese query parser fallback.
- [ ] Implement inclusive date overlap, alias-normalized disaster/city matching and evidence-image constraints.
- [ ] Re-run the test module and verify all assertions pass.

### Task 2: LangChain Parsing and Hybrid Retrieval

**Files:**
- Create: `backend/app/services/agent/case_multidim_search/vector_retriever.py`
- Create: `backend/app/services/agent/case_multidim_search/fusion.py`
- Modify: `backend/app/services/agent/case_multidim_search/query_parser.py`
- Test: `tests/case_multidim_search/test_hybrid_retrieval.py`

- [ ] Write tests with fake LangChain structured output, fake embeddings and vector chunk hits.
- [ ] Verify tests fail before vector and fusion implementations exist.
- [ ] Implement optional `ChatOpenAI.with_structured_output` parsing with deterministic fallback and warning metadata.
- [ ] Map vector chunks to `StandardCase` through `source_chunk_ids` first and `source_pdf` second.
- [ ] Implement reciprocal-rank-style fusion while retaining structured hard-filter eligibility.
- [ ] Verify hybrid, structured-only degradation, deduplication and score-order tests pass.

### Task 3: Traceable Intensity Extraction

**Files:**
- Create: `backend/app/services/agent/case_multidim_search/intensity.py`
- Test: `tests/case_multidim_search/test_intensity.py`

- [ ] Write failing tests for wind speed, precipitation, radar reflectivity and pressure values with units and source text.
- [ ] Write failing tests proving no metric is invented when source evidence is absent.
- [ ] Implement validated regex extraction from related chunks, case facts and image nearby text.
- [ ] Preserve metric name, numeric value, unit, location, relation, source ID, excerpt and confidence.
- [ ] Verify unspecified intensity requests return all available metrics and specified requests filter by metric name.

### Task 4: Aggregation, Charts, CSV and XLSX

**Files:**
- Create: `backend/app/services/agent/case_multidim_search/aggregator.py`
- Create: `backend/app/services/agent/case_multidim_search/chart_tool.py`
- Create: `backend/app/services/agent/case_multidim_search/exporter.py`
- Test: `tests/case_multidim_search/test_reporting_tools.py`

- [ ] Write failing tests for month/disaster/city/data-category aggregation and chart selection.
- [ ] Write failing tests for line, bar, histogram and scatter image generation with nonblank pixel checks.
- [ ] Implement whitelist-only `ChartSpec` generation; never execute model-produced plotting code.
- [ ] Implement Chinese-font Matplotlib chart rendering with preserved labels and units.
- [ ] Implement UTF-8 BOM CSV and a minimal valid XLSX writer with summary and case-detail sheets.
- [ ] Verify exported row counts and chart values equal the API aggregation data.

### Task 5: PDF, Agent and FastAPI Routes

**Files:**
- Create: `backend/app/services/agent/case_multidim_search/pdf_report.py`
- Create: `backend/app/services/agent/case_multidim_search/agent.py`
- Create: `backend/app/services/agent/case_multidim_search/router.py`
- Modify: `backend/app/main.py`
- Test: `tests/case_multidim_search/test_api_and_pdf.py`

- [ ] Write failing API tests for health, query, CSV, XLSX and PDF endpoints using injected temporary stores.
- [ ] Write failing PDF tests for Chinese text, page count, rendered nonblank pages, generated charts and evidence images.
- [ ] Implement the orchestration agent and service dependency container.
- [ ] Implement direct-download FastAPI responses with safe filenames and explicit degradation warnings.
- [ ] Render report sections for conditions, summary, aggregations, intensity, charts, case details, evidence images and data gaps.
- [ ] Register only the package router in `backend/app/main.py`.
- [ ] Verify Postman-equivalent HTTP tests pass.

### Task 6: Final Verification

**Files:**
- Test: `tests/case_multidim_search/`
- Modify: `requirements.txt` only if an imported runtime dependency is not already declared.

- [ ] Run `python -m unittest discover -s tests/case_multidim_search -v` and require zero failures.
- [ ] Run existing agent and standard-case unittest modules and record any external-service timeout separately.
- [ ] Run `python -m compileall -q backend tests/case_multidim_search`.
- [ ] Generate a report from repository data, render every page with pypdfium2, and verify nonblank pixels and image presence.
- [ ] Run `git diff --check` and confirm production code remains under the single agent directory except router registration.
