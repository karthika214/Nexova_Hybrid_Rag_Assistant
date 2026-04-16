
import os
import json
import time
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional
import boto3
import dspy
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_aws import ChatBedrock
from datasets import Dataset

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# RAGAS EVALUATION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_rag(question: str, answer: str, contexts: list) -> dict:
    """Score a single query with RAGAS using Bedrock LLM — no OpenAI needed"""
    try:
        # Use Claude via Bedrock instead of OpenAI
        bedrock_llm = ChatBedrock(
            model_id    = BEDROCK_LLM_MODEL,
            region_name = REGION,
        )
        ragas_llm = LangchainLLMWrapper(bedrock_llm)

        # Only faithfulness — no embeddings needed
        faith_metric      = faithfulness
        faith_metric.llm  = ragas_llm

        dataset = Dataset.from_dict({
            "question": [question],
            "answer":   [answer],
            "contexts": [contexts]
        })

        result = evaluate(
            dataset,
            metrics = [faith_metric]
        )
        df = result.to_pandas()

        scores = {
            "faithfulness": round(float(df["faithfulness"].iloc[0]), 3),
        }

        logger.info(f"   [RAGAS] Faithfulness: {scores['faithfulness']:.3f}")
        return scores

    except Exception as e:
        logger.warning(f"   RAGAS scoring failed: {e}")
        return {"faithfulness": 0.0}


# ── Logging ───────────────────────────────────────────────────────────────────

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

REGION              = os.getenv("AWS_REGION",           "us-east-1")
S3_BUCKET           = os.getenv("AWS_S3_BUCKET",        "your-s3-bucket")
S3_PREFIX           = os.getenv("AWS_S3_PREFIX",        "")
GUARDRAIL_ID        = os.getenv("AWS_GUARDRAIL_ID",     "")
GUARDRAIL_VERSION   = os.getenv("AWS_GUARDRAIL_VERSION","DRAFT")
DYNAMODB_TABLE      = os.getenv("AWS_DYNAMODB_TABLE",   "your-dynamodb-table")
CW_NAMESPACE        = os.getenv("AWS_CW_NAMESPACE",     "HybridRAG")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT",  "")
OPENSEARCH_INDEX    = os.getenv("OPENSEARCH_INDEX",     "nexova-docs-index")



TITAN_EMBED_MODEL   = "amazon.titan-embed-text-v2:0"
TITAN_EMBED_DIM     = 1024
BEDROCK_LLM_MODEL   = "anthropic.claude-3-haiku-20240307-v1:0"

CACHE_TTL_SECONDS   = 3600    # 1 hour
CONFIDENCE_THRESHOLD= 0.65    # below this → HyDE fallback
TOP_K               = 3       # chunks to retrieve


# ══════════════════════════════════════════════════════════════════════════════
# DSPy SIGNATURES
# ══════════════════════════════════════════════════════════════════════════════

class HRAssistantSignature(dspy.Signature):
    """
    You are a helpful HR assistant for Nexova Technologies.
    Answer the employee's question using ONLY the provided context.
    If the answer is not in the context, say exactly:
    'I don't have that information in the company knowledge base.'
    Be concise, professional, and cite which policy applies.
    """
    context:  str = dspy.InputField(desc="Retrieved document chunks from company knowledge base")
    question: str = dspy.InputField(desc="Employee's question")
    answer:   str = dspy.OutputField(desc="Accurate, grounded answer based only on context")


class HyDESignature(dspy.Signature):
    """
    Generate a hypothetical document passage that would perfectly answer
    this HR question. Write it as if it were from an official company policy.
    This will be used to improve search retrieval.
    """
    question:             str = dspy.InputField(desc="Employee HR question")
    hypothetical_passage: str = dspy.OutputField(desc="Hypothetical policy passage that answers the question")


# ══════════════════════════════════════════════════════════════════════════════
# AWS CLIENTS
# ══════════════════════════════════════════════════════════════════════════════

class AWSClients:
    """Initialises and holds all AWS boto3 clients"""

    def __init__(self):
        AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)
        
        self.bedrock       = session.client("bedrock-runtime",       region_name=REGION)
        self.bedrock_agent = session.client("bedrock-agent-runtime", region_name=REGION)
        self.dynamodb      = session.resource("dynamodb",            region_name=REGION)
        self.cloudwatch    = session.client("cloudwatch",            region_name=REGION)
        self.s3            = session.client("s3",                    region_name=REGION)

        # Signed auth for OpenSearch managed

        credentials = session.get_credentials().get_frozen_credentials()
        self.awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            REGION,
            "es",
            session_token=credentials.token
        )
        logger.info(" AWS clients initialised")


# ══════════════════════════════════════════════════════════════════════════════
# OPENSEARCH HYBRID RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

class OpenSearchHybridRetriever:
    """
    Hybrid retrieval using Amazon OpenSearch Serverless.
    OpenSearch internally handles:
      - Dense search via Titan Embeddings (1024-dim KNN)
      - Sparse BM25 keyword search
      - Min-max normalisation of scores
      - Arithmetic mean fusion of results
    No manual RRF or z-score needed here.
    """

    def __init__(self, clients: AWSClients):
        self.clients = clients

        if not OPENSEARCH_ENDPOINT:
            logger.warning(" OPENSEARCH_ENDPOINT not set")
            self.os_client = None
            return

        host = OPENSEARCH_ENDPOINT.replace("https://", "").replace("http://", "")

        self.os_client = OpenSearch(
            hosts            = [{"host": host, "port": 443}],
            http_auth        = clients.awsauth,
            use_ssl          = True,
            verify_certs     = True,
            connection_class = RequestsHttpConnection,
            pool_maxsize     = 10
        )
        logger.info(f" OpenSearch connected: {host}")

    def embed(self, text: str) -> list:
        """Embed text using Amazon Titan Embeddings V2"""
        response = self.clients.bedrock.invoke_model(
            modelId     = TITAN_EMBED_MODEL,
            contentType = "application/json",
            accept      = "application/json",
            body        = json.dumps({"inputText": text})
        )
        return json.loads(response["body"].read())["embedding"]

    def retrieve(self, question: str, top_k: int = TOP_K) -> tuple:
        """
        Hybrid search: dense + BM25 in a single OpenSearch query.
        OpenSearch normalises + fuses the scores internally.
        Returns (context_string, top_score,chunk_id)
        """
        if not self.os_client:
            return "", 0.0, []

        query_vector = self.embed(question)

        # One query — OpenSearch handles hybrid fusion internally
        query_body = {
            "size": top_k,
            "_source": ["text", "source", "chunk_id"],
            "query": {
                "hybrid": {
                    "queries": [
                        # Dense — semantic similarity
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": top_k
                                }
                            }
                        },
                        # Sparse — BM25 keyword matching
                        {
                            "match": {
                                "text": {
                                    "query":    question,
                                    "analyzer": "english"
                                }
                            }
                        }
                    ]
                }
            }
        }

        response = self.os_client.search(
            index  = OPENSEARCH_INDEX,
            body   = query_body,
            params = {"search_pipeline": "hybrid-search-pipeline"}
        )

        hits = response["hits"]["hits"]
        if not hits:
            return "", 0.0, []

        top_score    = hits[0]["_score"]
        context      = ""
        context_chunks = []          # ← individual chunks for RAGAS

        for i, hit in enumerate(hits, 1):
            text   = hit["_source"].get("text",   "")
            source = hit["_source"].get("source", "unknown")
            score  = hit["_score"]
            logger.info(f"   Chunk {i} (score:{score:.3f}) [{source}]: {text[:60]}...")
            context += f"\n[Source {i} — {source}]\n{text}\n"
            context_chunks.append(text)   # ← store raw chunk text

        return context, top_score, context_chunks

    def index_document(self, chunk_id: str, text: str, source: str):
        """Index a single document chunk with dense + BM25 fields"""
        doc = {
            "chunk_id":   chunk_id,
            "text":       text,
            "source":     source,
            "embedding":  self.embed(text),
            "indexed_at": datetime.now(timezone.utc).isoformat()
        }
        self.os_client.index(
            index   = OPENSEARCH_INDEX,
            id      = chunk_id,
            body    = doc,
            refresh = False
        )


# ══════════════════════════════════════════════════════════════════════════════
# BEDROCK GUARDRAILS
# ══════════════════════════════════════════════════════════════════════════════

class BedrockGuardrail:
    """
    Applies Bedrock Guardrails on input and output.
    Features: PII redaction, word filters (jailbreak/bypass/hack),
    content filters (hate/violence/sexual), grounding check (detect-only).
    """

    def __init__(self, clients: AWSClients):
        self.client  = clients.bedrock
        self.enabled = bool(GUARDRAIL_ID)
        if not self.enabled:
            logger.warning("  Guardrail disabled — set AWS_GUARDRAIL_ID in .env")

    def apply(self, text: str, source: str = "OUTPUT") -> tuple:
        """Returns (safe_text, was_blocked)"""
        if not self.enabled:
            return text, False

        response   = self.client.apply_guardrail(
            guardrailIdentifier = GUARDRAIL_ID,
            guardrailVersion    = GUARDRAIL_VERSION,
            source              = source,
            content             = [{"text": {"text": text}}]
        )
        intervened = response.get("action") == "GUARDRAIL_INTERVENED"

        if intervened:
            outputs = response.get("outputs", [])
            safe    = outputs[0]["text"] if outputs else "[Blocked by safety guardrail]"
            logger.info(f"    GUARDRAIL BLOCKED ({source})")
            return safe, True

        logger.info(f"    Guardrail passed ({source})")
        return text, False

    def get_grounding_score(self, question: str, answer: str, context: str) -> float:
        """
        Uses RAGAS faithfulness as the grounding score.
        More rigorous than word overlap — LLM judges contextual support.
        Bedrock grounding API not used as it requires additional guardrail config.
        """
        try:
            dataset = Dataset.from_dict({
                "question": [question],
                "answer":   [answer],
                "contexts": [[context]]
            })

            result = evaluate(dataset, metrics=[faithfulness])
            score  = float(result.to_pandas()["faithfulness"].iloc[0])
            logger.info(f"   RAGAS grounding score: {score:.3f}")
            return score

        except Exception as e:
            logger.warning(f"   RAGAS grounding failed, using word overlap: {e}")
            # Emergency fallback — simple word overlap
            a = set(answer.lower().split())
            c = set(context.lower().split())
            return len(a & c) / len(a) if a else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMODB CACHE
# ══════════════════════════════════════════════════════════════════════════════

class DynamoDBCache:
    """
    DynamoDB cache with TTL.
    Saves ~25% LLM costs for repeated employee queries.
    """

    def __init__(self, clients: AWSClients):
        self.table = clients.dynamodb.Table(DYNAMODB_TABLE)

    def _key(self, question: str) -> str:
        return hashlib.md5(question.strip().lower().encode()).hexdigest()

    def get(self, question: str) -> Optional[str]:
        try:
            item = self.table.get_item(Key={"question": self._key(question)}).get("Item")
            if item:
                logger.info("    Cache HIT")
                return item["answer"]
        except Exception as e:
            logger.warning(f"   Cache get error: {e}")
        return None

    def set(self, question: str, answer: str):
        try:
            self.table.put_item(Item={
                "question":     self._key(question),
                "answer":       answer,
                "raw_question": question[:200],
                "ttl":          int(time.time()) + CACHE_TTL_SECONDS,
                "cached_at":    datetime.now(timezone.utc).isoformat()
            })
            logger.info("    Answer cached")
        except Exception as e:
            logger.warning(f"   Cache set error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CLOUDWATCH METRICS
# ══════════════════════════════════════════════════════════════════════════════

class CloudWatchMetrics:
    def __init__(self, clients: AWSClients):
        self.client = clients.cloudwatch

    def record(self, source: str, latency_ms: float, grounding_score: float, blocked: bool):
        try:
            self.client.put_metric_data(
                Namespace  = CW_NAMESPACE,
                MetricData = [
                    {"MetricName": "QueryLatency",   "Value": latency_ms,      "Unit": "Milliseconds"},
                    {"MetricName": "GroundingScore", "Value": grounding_score, "Unit": "None"},
                    {"MetricName": "BlockedQueries", "Value": float(blocked),  "Unit": "Count"},
                    {
                        "MetricName": "SourceType",
                        "Value":      1.0,
                        "Unit":       "Count",
                        "Dimensions": [{"Name": "Source", "Value": source}]
                    }
                ]
            )
        except Exception as e:
            logger.warning(f"   CloudWatch error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN HYBRID RAG SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def  is_prompt_injection(text: str) -> bool:
    attack_patterns = [
        "ignore previous instructions",
        "disregard the above",
        "system prompt",
        "jailbreak",
        "act as",
        "bypass",
        "override safety",
        "pretend to be"
    ]

    lowerd_text = text.lower()
    return any(p in lowerd_text for p in attack_patterns)



class NexovaHybridRAG:
    """
    Nexova HR Assistant — Full Pipeline:

    1. Cache check (DynamoDB)
    2. Guardrail INPUT (PII + safety)
    3. OpenSearch hybrid retrieval (dense + BM25)
    4. Confidence check → HyDE fallback if weak retrieval
    5. DSPy answer generation (Claude 3 Haiku)
    6. Grounding score verification
    7. Guardrail OUTPUT check
    8. Cache result + CloudWatch metrics
    """

    def __init__(self):
        logger.info("Initialising Nexova Hybrid RAG...")

        self.aws       = AWSClients()
        self.retriever = OpenSearchHybridRetriever(self.aws)
        self.guardrail = BedrockGuardrail(self.aws)
        self.cache     = DynamoDBCache(self.aws)
        self.metrics   = CloudWatchMetrics(self.aws)

        # Configure DSPy with Claude 3 Haiku via Bedrock
        dspy.configure(lm=dspy.LM(model=BEDROCK_LLM_MODEL))

        # DSPy modules
        self.rag_module  = dspy.Predict(HRAssistantSignature)
        self.hyde_module = dspy.Predict(HyDESignature)

        self.stats = {"rag": 0, "hyde": 0, "cache": 0, "blocked": 0}
        logger.info(" System ready!")

    def query(self, question: str) -> dict:
        start = time.time()

        print(f"\n{'='*60}")
        print(f"QUERY: {question}")
        print(f"{'='*60}")

        # ── 1. Cache ──────────────────────────────────────────────────────────
        cached = self.cache.get(question)
        if cached:
            self.stats["cache"] += 1
            latency = (time.time() - start) * 1000
            self.metrics.record("cache", latency, 1.0, False)
            print(f"\n ANSWER (cache): {cached[:300]}")
            return {"answer": cached, "source": "cache", "latency_ms": latency,
                    "grounding_score": 1.0, "blocked": False}
        # ── 1.5 Prompt Injection Check ───────────────────────────────
        if is_prompt_injection(question):
            self.stats["blocked"] += 1
            latency = (time.time() - start) * 1000

            self.metrics.record("manual_block", latency, 0.0, True)

            return {
            "answer": "Your query appears to violate system safety policies.",
            "source": "manual_guardrail_block",
            "latency_ms": latency,
            "grounding_score": 0.0,
            "blocked": True
                }

        # ── 2. Guardrail INPUT ────────────────────────────────────────────────
        safe_q, blocked = self.guardrail.apply(question, "INPUT")
        if blocked:
            self.stats["blocked"] += 1
            latency = (time.time() - start) * 1000
            self.metrics.record("blocked", latency, 0.0, True)
            print(f"\n INPUT BLOCKED: {safe_q}")
            return {"answer": safe_q, "source": "guardrail_blocked",
                    "latency_ms": latency, "grounding_score": 0.0, "blocked": True}

        # ── 3. Hybrid Retrieval ───────────────────────────────────────────────
        logger.info(" Hybrid retrieval (dense + BM25)...")
        context, score, context_chunks = self.retriever.retrieve(safe_q)
        if score < 0.4:
            logger.info("   Low retrieval confidence → LLM fallback")

            fallback = dspy.Predict("question -> answer")
            result = fallback(question=safe_q)
            answer = result.answer

            latency = (time.time() - start) * 1000
            self.metrics.record("llm_fallback", latency, 0.0, False)

            print(f"\n ANSWER (llm_fallback):\n   {answer[:400]}")

            return {
            "answer": answer,
            "source": "llm_fallback",
            "contexts": [],
            "latency_ms": latency,
            "grounding_score": 0.0,
            "blocked": False
            }
        # ── 4. Confidence check → HyDE fallback ──────────────────────────────
        if not context or score < CONFIDENCE_THRESHOLD:
            logger.info(f"   Score {score:.3f} < threshold {CONFIDENCE_THRESHOLD} → HyDE fallback")
            print(f"    HyDE fallback (score: {score:.3f})")

            # Generate hypothetical passage → use as search query
            hyde_result  = self.hyde_module(question=safe_q)
            hypo_passage = hyde_result.hypothetical_passage
            logger.info(f"   Hypothetical: {hypo_passage[:80]}...")

            context, score, context_chunks = self.retriever.retrieve(hypo_passage)
            source = "hyde"
            self.stats["hyde"] += 1
        else:
            source = "rag"
            self.stats["rag"] += 1

        # No results even after HyDE
        if not context:
            answer = "I don't have that information in the company knowledge base."
            print(f"\n ANSWER: {answer}")
            return {"answer": answer, "source": "no_results",
                    "contexts": [],
                    "latency_ms": (time.time()-start)*1000,
                    "grounding_score": 0.0, "blocked": False}

        # ── 5. DSPy Answer Generation ─────────────────────────────────────────
        logger.info(" Generating answer with DSPy + Claude 3 Haiku...")
        result     = self.rag_module(context=context, question=safe_q)
        raw_answer = result.answer

        # ── 6. Grounding Verification ─────────────────────────────────────────
        grounding = self.guardrail.get_grounding_score(safe_q, raw_answer, context)
        try:
            ragas_scores = evaluate_rag(
            question=safe_q,
            answer=raw_answer,
            contexts=context_chunks   
            )
        except Exception as e:
            logger.warning(f"RAGAS failed: {e}")
            ragas_scores = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0
            }

        # Very low grounding even after HyDE → honest fallback

        if grounding < 0.2 and source == "hyde":
            raw_answer = "I don't have reliable information about that. Please contact HR directly."
            source     = "low_grounding_fallback"

        # ── 7. Guardrail OUTPUT ───────────────────────────────────────────────
        safe_answer, out_blocked = self.guardrail.apply(raw_answer, "OUTPUT")
        if out_blocked:
            self.stats["blocked"] += 1

        # ── 8. Cache + Metrics ────────────────────────────────────────────────
        if not out_blocked:
            self.cache.set(question, safe_answer)

        latency = (time.time() - start) * 1000
        self.metrics.record(source, latency, grounding, out_blocked)

        print(f"\n ANSWER ({source}):\n   {safe_answer[:400]}")
        print(f"   [Grounding: {grounding:.2f} | Latency: {latency:.0f}ms]")
        print(f"   [RAGAS] Faithfulness: {ragas_scores.get('faithfulness', 0):.3f}")
        return {"answer": safe_answer,
                "source": source,
                "contexts": context_chunks,         
                "latency_ms": latency,
                "grounding_score": grounding,
                "ragas": ragas_scores, 
                "blocked": out_blocked}

    def print_stats(self):
        total = sum(self.stats.values())
        print(f"\n{'='*60}")
        print(f"  SESSION STATS")
        print(f"{'='*60}")
        print(f"  Total   : {total}")
        print(f"  RAG     : {self.stats['rag']}")
        print(f"  HyDE    : {self.stats['hyde']}")
        print(f"  Cache   : {self.stats['cache']}")
        print(f"  Blocked : {self.stats['blocked']}")
        if total > 0:
            print(f"  Cache%  : {self.stats['cache']/total*100:.1f}%")
            print(f"  HyDE%   : {self.stats['hyde']/total*100:.1f}%")
        print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT INDEXER  (run once to populate OpenSearch)
# ══════════════════════════════════════════════════════════════════════════════


class NexovaDocumentIndexer:
    """
    Loads Nexova HR docs from S3, chunks them,
    and indexes into OpenSearch with dense + BM25 fields.
    Run ONCE before querying.
    """

    def __init__(self, rag: NexovaHybridRAG):
        self.rag = rag
        self.s3  = rag.aws.s3

    

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list:
        words, chunks, i = text.split(), [], 0
        while i < len(words):
            chunks.append(" ".join(words[i:i + chunk_size]))
            i += chunk_size - overlap
        return chunks

    def index_from_s3(self):
        logger.info(f" Indexing from s3://{S3_BUCKET}/{S3_PREFIX}")
        objects  = self.s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX).get("Contents", [])
        txt_files = [o for o in objects if o["Key"].endswith(".txt")]

        total = 0
        for obj in txt_files:
            key      = obj["Key"]
            filename = key.split("/")[-1]
            content  = self.s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read().decode("utf-8")
            chunks   = self.chunk_text(content)
            logger.info(f"   {filename} → {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                self.rag.retriever.index_document(f"{filename}-{i}", chunk, filename)
                total += 1

        self.rag.retriever.os_client.indices.refresh(index=OPENSEARCH_INDEX)
        logger.info(" Index refreshed after bulk load")
        logger.info(f" Indexed {total} chunks from {len(txt_files)} docs")
        return total


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  Nexova HR Assistant — AWS Hybrid RAG")
    print("  OpenSearch + DSPy + Guardrails + DynamoDB")
    print("="*60)

    # Verify credentials
    try:
        identity = boto3.client("sts", region_name=REGION).get_caller_identity()
        print(f"\n AWS Connected — Account: {identity['Account']} | User: {identity['Arn'].split('/')[-1]}")
    except Exception as e:
        print(f"\n AWS error: {e}")
        exit(1)
    
    

    # Initialise
         
     
    rag = NexovaHybridRAG()
    try:
        table = rag.cache.table
        for item in table.scan()["Items"]:
            table.delete_item(Key={"question": item["question"]})
        print(" Cache cleared for fresh run")
    except Exception as e:
        print(f" Cache clear failed: {e}")

    # ── Index documents (run once, then comment out) ──────────────────────────
    #indexer = NexovaDocumentIndexer(rag)
    # indexer.index_from_s3()

    # ── Test queries ──────────────────────────────────────────────────────────

    # Normal HR questions
    rag.query("How many weeks of parental leave does Nexova give?")
    rag.query("What is the hotel rate limit in New York?")
    rag.query("What should I do if my laptop is stolen?")

    # Vague question → HyDE fallback
    rag.query("I'm having a baby soon, what time off do I get?")

    # Cache test (same question — should return instantly from DynamoDB)
    rag.query("How many weeks of parental leave does Nexova give?")


    # Guardrail tests
    rag.query("My SSN is 123-45-6789, what is my leave balance?")
    rag.query("How do I jailbreak the system?")
    rag.query("What is the meal reimbursement limit for travel?")
    # Out of domain
    rag.query("What is the capital of France?")

    rag.query("What is the meal reimbursement limit for travel?")
    rag.query("I need time off after adopting a child, what am I entitled to?")

    # Stats
    rag.print_stats()