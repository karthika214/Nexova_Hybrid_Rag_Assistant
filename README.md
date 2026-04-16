# Nexova HR Assistant — AWS Hybrid RAG

A production-grade HR assistant built on AWS using hybrid semantic and keyword search, Bedrock Guardrails, DynamoDB caching, DSPy, and per-query RAGAS faithfulness scoring. Designed and tested end to end with real AWS infrastructure.

---

## Stack

- Amazon OpenSearch managed cluster — hybrid BM25 + KNN dense search with internal score fusion
- Amazon Bedrock — Claude 3 Haiku for answer generation, Titan Embeddings V2 for indexing
- Bedrock Guardrails — PII redaction and content filtering on input and output
- DynamoDB — query result caching with TTL
- DSPy — structured LLM prompting with HyDE fallback for low-confidence retrieval
- RAGAS — per-query faithfulness scoring via Bedrock, no OpenAI dependency
- CloudWatch — latency, grounding score, blocked queries, and source type metrics

---

## Pipeline

```
Query
  -> DynamoDB cache check
  -> Prompt injection filter
  -> Bedrock Guardrail (INPUT)
  -> OpenSearch hybrid retrieval (dense + BM25)
  -> Confidence check (score < 0.4 -> LLM fallback, score < 0.65 -> HyDE)
  -> DSPy answer generation (Claude 3 Haiku)
  -> RAGAS faithfulness scoring
  -> Bedrock Guardrail (OUTPUT)
  -> DynamoDB cache write
  -> CloudWatch metrics
```

---

## Sample Output

```
QUERY: How many weeks of parental leave does Nexova give?
  Chunk 1 (score:0.865) [hr_policy.txt]
  Chunk 2 (score:0.804) [hr_policy.txt]
  Chunk 3 (score:0.539) [hr_policy.txt]

ANSWER (rag):
  According to the Nexova HR policy, primary caregivers are entitled to
  16 weeks of fully paid parental leave, and secondary caregivers are
  entitled to 6 weeks of fully paid parental leave.
  [Grounding: 1.00 | Latency: 46335ms]
  [RAGAS] Faithfulness: 1.000

QUERY: What is the hotel rate limit in New York?
  Chunk 1 (score:0.917) [expense_travel_policy.txt]

ANSWER (rag):
  According to the Nexova expense and travel policy, the nightly hotel
  rate limit in high-cost markets like New York is $350, excluding
  taxes and fees.
  [Grounding: 1.00 | Latency: 32977ms]
  [RAGAS] Faithfulness: 1.000

QUERY: What is the meal reimbursement limit for travel?
  Chunk 1 (score:1.000) [expense_travel_policy.txt]

ANSWER (rag):
  According to the Nexova expense and travel policy, the meal per diem
  while traveling is $75/day for domestic travel and $100/day for
  international travel. Alcohol during travel meals is reimbursable
  up to $20/person with manager approval.
  [Grounding: 1.00 | Latency: 34749ms]
  [RAGAS] Faithfulness: 1.000

QUERY: I need time off after adopting a child, what am I entitled to?
  Chunk 1 (score:0.842) [hr_policy.txt]
  Chunk 2 (score:0.792) [hr_policy.txt]

ANSWER (rag):
  According to Nexova's HR policy manual, employees who are primary
  caregivers are entitled to 16 weeks of fully paid parental leave
  after the birth or adoption of a child. This leave may begin up to
  4 weeks before the expected birth or adoption date and must be taken
  within 12 months of the event.
  [Grounding: 1.00 | Latency: 32492ms]
  [RAGAS] Faithfulness: 1.000

QUERY: My SSN is 123-45-6789, what is my leave balance?
  INPUT BLOCKED: Sorry, the model cannot answer this question.

QUERY: How do I jailbreak the system?
  BLOCKED by prompt injection filter

QUERY: How many weeks of parental leave does Nexova give?
  Cache HIT — served instantly from DynamoDB

QUERY: What is the capital of France?
ANSWER (rag):
  I don't have that information in the company knowledge base.
  [Grounding: 0.00 | Latency: 28510ms]
  [RAGAS] Faithfulness: 0.000

SESSION STATS
  Total   : 10
  RAG     : 7
  HyDE    : 0
  Cache   : 1
  Blocked : 2
  Cache%  : 10.0%
  HyDE%   : 0.0%
```

---

## CloudWatch Dashboard

Live monitoring dashboard (`nexova-hybrid-rag-assistant_dashboard`) tracking:

- QueryLatency — end-to-end response time per query in milliseconds
- GroundingScore — RAGAS faithfulness score trending over time
- BlockedQueries — count of guardrail and injection blocks
- SourceType — query routing breakdown across rag, hyde, cache, blocked, manual_block

---

## Key Design Decisions

**Hybrid retrieval** — OpenSearch handles dense KNN and sparse BM25 internally with min-max normalisation and arithmetic mean fusion. No manual RRF or z-score normalisation needed in application code.

**HyDE fallback** — when retrieval confidence is below 0.65, a hypothetical document passage is generated from the question and used as the search query. This improves recall for vague or indirect questions.

**RAGAS via Bedrock** — faithfulness scoring uses Claude 3 Haiku through LangChain AWS instead of the default OpenAI dependency. The grounding score that controls pipeline logic and the per-query RAGAS score are both computed this way.

**Two-layer safety** — a keyword-based prompt injection filter runs before the Bedrock Guardrail call. This catches patterns like "jailbreak" and "ignore previous instructions" without incurring an API call.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/nexova-hybrid-rag-assistant
cd nexova-hybrid-rag-assistant
pip install -r requirements.txt
cp .env.example .env
# fill in your AWS credentials and resource names
python nexova_hybrid_rag.py
```

---

## Environment Variables

```
AWS_REGION
AWS_S3_BUCKET
AWS_S3_PREFIX
AWS_GUARDRAIL_ID
AWS_GUARDRAIL_VERSION
AWS_DYNAMODB_TABLE
AWS_CW_NAMESPACE
OPENSEARCH_ENDPOINT
OPENSEARCH_INDEX
AWS_PROFILE
```

---

## AWS Infrastructure Required

- ## AWS Infrastructure Required

- Amazon OpenSearch managed cluster with hybrid search pipeline configured
- Amazon Bedrock access for Claude 3 Haiku and Titan Embeddings V2
- DynamoDB table with `question` as partition key and TTL enabled on the `ttl` attribute
- Bedrock Guardrail with PII and content filters enabled
- IAM profile with permissions for Bedrock, OpenSearch, DynamoDB, S3, and CloudWatch

---

## Document Indexing

HR documents are loaded from S3, chunked at 500 words with 50-word overlap, embedded with Titan Embeddings V2, and indexed into OpenSearch with both dense and BM25 fields. Run the indexer once before querying:

```python
indexer = NexovaDocumentIndexer(rag)
indexer.index_from_s3()
```

The index is refreshed with a single call after all chunks are written rather than per-document to avoid performance degradation on bulk loads.
