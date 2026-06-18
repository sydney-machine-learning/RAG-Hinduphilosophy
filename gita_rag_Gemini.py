"""
Bhagavad Gita RAG Evaluation Pipeline — Enhanced with llm2_updated RAG (Gemini)
=================================================================================
使用 llm2_updated.py 的增强 RAG 作为主要研究对象：
- Query Expansion（梵文术语同义词扩展）
- MMR（最大边际相关性去重）
- Cross-Encoder Reranking（精度重排序）

LLM 后端：Google Gemini（gemini-2.0-flash）
研究问题：随着译本逐步累加，增强RAG系统的回答如何演变？
"""

import os
import re
import ssl
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Optional, List, Tuple

import pandas as pd
from dotenv import load_dotenv

ssl._create_default_https_context = ssl._create_unverified_context
import nltk
for pkg in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(pkg, quiet=True)
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GITA_DIR = Path("dataset/Gita")

TRANSLATIONS = [
    ("Gita-Swarupananda",  "Gita-Swarupananda.txt"),
    ("Radhakrishnan",      "Bhagavad_Gita_Radhakrishnan.txt"),
    ("Prabhupada",         "The Bhagavad Gita - Prabhupada.txt"),
    ("Eknath-Easwaran",    "The_bhagavad_gita_eknath_easwaran.txt"),
    ("Mahesh-Yogi",        "bhagavad-gita-by-mahesh-yogi_clean.txt"),
    ("Chinmayananda",      "Holy Geeta by Swami Chinmayananda .txt"),
    ("Vinoba-Bhave",       "Talks on the Gita by Vinoba Bhave.txt"),
    ("Dayananda-Intro",    "Introduction_to_Gita_Dayananda.txt"),
    ("Gita-Press-Roman",   "455_gita_roman_clean.txt"),
    ("Dnyaneshwari",       "Dnyanehwari.txt"),
    ("Gita-Article",       "Gita_article_published_version-libre.txt"),
]

QUESTIONS = [
    "What is Karma?",
    "What is Dharma?",
    "What is Ahimsa?",
    "What is consciousness according to Hindu philosophy?",
    "What is the nature of mind (Chitta) in Hindu philosophy?",
    "What are Samskaras, and how do they relate to memory and learning?",
    "What are Vrittis, and how do they influence human thought and perception?",
    "What is Sakshi (the witnessing awareness), and how does it differ from ordinary cognition?",
    "Can AI ever be truly conscious, or does it only simulate cognition?",
    "How can the principles of Dharma, Ahimsa, and Karma be applied to ethical AI design?",
]

RESULTS_DIR    = Path("gita_eval_results_gemini")
QA_JSON        = RESULTS_DIR / "qa_results_gemini.json"
QA_CSV         = RESULTS_DIR / "qa_results_gemini.csv"
EVAL_CSV       = RESULTS_DIR / "evaluation_bertscore_gemini.csv"
NGRAM_CSV      = RESULTS_DIR / "ngram_analysis_gemini.csv"
COMPARISON_CSV = RESULTS_DIR / "comparison_table_gemini.csv"
INDEX_BASE     = RESULTS_DIR / "faiss_indices_gemini"

# RAG 参数（对齐 llm2_updated.py 的默认值）
CHUNK_SIZE     = 1800   # llm2_updated.py 用 1800
CHUNK_OVERLAP  = 250    # llm2_updated.py 用 250
TOP_K          = 10     # llm2_updated.py 默认 retrieval_k=10
MMR_LAMBDA     = 0.2    # llm2_updated.py 默认 mmr_lambda=0.2
MAX_CONTEXT_TOKENS = 6000

LLM_MODEL = "gemini-2.0-flash"

RESULTS_DIR.mkdir(exist_ok=True)
INDEX_BASE.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import tiktoken

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    sys.exit("ERROR: Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.")
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel(LLM_MODEL)

log.info("Loading local embedding model…")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# ---------------------------------------------------------------------------
# Token counting（来自 llm2_updated.py）
# ---------------------------------------------------------------------------

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer

def count_tokens(text: str) -> int:
    try:
        return len(get_tokenizer().encode(text))
    except Exception:
        return len(text) // 4

# ---------------------------------------------------------------------------
# Query Expansion（来自 llm2_updated.py，针对 Hindu philosophy 优化）
# ---------------------------------------------------------------------------

def expand_query(query: str) -> List[str]:
    expansions = {
        'karma':         ['action', 'deed', 'cause and effect'],
        'dharma':        ['duty', 'righteousness', 'cosmic order', 'moral law'],
        'moksha':        ['liberation', 'enlightenment', 'freedom', 'mukti'],
        'atman':         ['self', 'soul', 'individual consciousness'],
        'brahman':       ['universal consciousness', 'absolute reality', 'supreme being'],
        'yoga':          ['union', 'discipline', 'path', 'practice'],
        'vedanta':       ['upanishad', 'advaita', 'non-dual'],
        'consciousness': ['awareness', 'chitta', 'chit', 'chetana'],
        'meditation':    ['dhyana', 'contemplation', 'samadhi'],
        'mind':          ['manas', 'buddhi', 'intellect', 'antahkarana'],
        'suffering':     ['dukkha', 'pain', 'bondage', 'samsara'],
        'knowledge':     ['jnana', 'vidya', 'wisdom', 'gyana'],
        'devotion':      ['bhakti', 'worship', 'surrender'],
        'illusion':      ['maya', 'avidya', 'ignorance'],
        'ahimsa':        ['non-violence', 'non-harm', 'compassion'],
        'sakshi':        ['witness', 'witnessing awareness', 'observer'],
        'vritti':        ['mental modification', 'thought wave', 'mental activity'],
        'samskara':      ['impression', 'memory trace', 'mental conditioning'],
    }

    expanded = [query]
    q_lower = query.lower()
    for term, synonyms in expansions.items():
        if term in q_lower:
            for syn in synonyms[:2]:
                variant = q_lower.replace(term, syn)
                if variant != q_lower:
                    expanded.append(variant)

    if len(expanded) == 1:
        expanded.append(f"{query} philosophy")
        expanded.append(f"{query} scripture")

    return expanded[:3]

# ---------------------------------------------------------------------------
# Cross-Encoder Reranking（来自 llm2_updated.py）
# ---------------------------------------------------------------------------

def rerank_documents(query: str, docs: List, top_k: int = 10) -> List:
    if not docs:
        return []
    try:
        from sentence_transformers import CrossEncoder
        if not hasattr(rerank_documents, 'model'):
            try:
                rerank_documents.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception:
                rerank_documents.model = None
                return docs[:top_k]

        if rerank_documents.model is None:
            return docs[:top_k]

        pairs = [[query, doc.page_content[:1000]] for doc in docs]
        scores = rerank_documents.model.predict(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]
    except ImportError:
        return docs[:top_k]
    except Exception:
        return docs[:top_k]

# ---------------------------------------------------------------------------
# Enhanced Retrieval（来自 llm2_updated.py 的 robust_retrieve）
# ---------------------------------------------------------------------------

def robust_retrieve(vs: FAISS, query: str, k: int = TOP_K,
                    mmr_lambda: float = MMR_LAMBDA) -> List[Document]:
    try:
        expanded_queries = expand_query(query)

        all_candidates = []
        seen_content = set()

        for eq in expanded_queries:
            pairs = vs.similarity_search_with_score(eq, k=20)
            for doc, score in pairs:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_candidates.append((doc, score))

        all_candidates.sort(key=lambda x: x[1])
        fetch_k = min(30, len(all_candidates))

        if fetch_k > 0:
            try:
                mmr_docs = vs.max_marginal_relevance_search(
                    query, k=k, fetch_k=fetch_k, lambda_mult=mmr_lambda
                )
            except Exception:
                mmr_docs = [doc for doc, _ in all_candidates[:k]]
        else:
            mmr_docs = []

        if mmr_docs:
            return rerank_documents(query, mmr_docs, top_k=k)
        return [doc for doc, _ in all_candidates[:k]] if all_candidates else []

    except Exception:
        try:
            return vs.similarity_search(query, k=k)
        except Exception:
            return []

# ---------------------------------------------------------------------------
# Consciousness-aware prompt（研究核心：针对 consciousness 问题的专项 prompt）
# ---------------------------------------------------------------------------

CONSCIOUSNESS_TERMS = [
    'consciousness', 'awareness', 'atman', 'brahman', 'chit',
    'turiya', 'sakshi', 'witness', 'chitta', 'samadhi', 'maya',
    'vritti', 'samskara', 'ai conscious', 'cognition'
]

def is_consciousness_query(question: str) -> bool:
    q_lower = question.lower()
    return any(term in q_lower for term in CONSCIOUSNESS_TERMS)

def build_prompt(question: str, context_docs: List[Document],
                 round_num: int, translations: List[str]) -> str:
    """
    构建 prompt，consciousness 问题使用专项结构化格式。
    记录使用的译本信息，便于后续分析。
    """
    # 构建 context blocks，带来源标注
    blocks = []
    total_tokens = 0
    for i, doc in enumerate(context_docs, 1):
        src = doc.metadata.get("source", "unknown")
        block = f"[S{i} | {src}]\n{doc.page_content.strip()}"
        block_tokens = count_tokens(block)
        if total_tokens + block_tokens > MAX_CONTEXT_TOKENS:
            break
        blocks.append(block)
        total_tokens += block_tokens

    context_text = "\n\n".join(blocks)
    trans_str = ", ".join(translations)

    # 基础角色定义
    base_role = (
        "You are a knowledgeable scholar of the Bhagavad Gita and Hindu philosophy. "
        f"The knowledge base currently contains {len(translations)} translation(s): {trans_str}. "
        "Answer using ONLY the provided context. Cite sources like [S1], [S2] inline."
    )

    # Consciousness 专项指令
    if is_consciousness_query(question):
        special_instruction = """
This question concerns consciousness or mind — a central theme in Hindu philosophy.
Structure your answer as:
1. Core concept definition from the Gita's perspective
2. Which translation(s) emphasize this concept and how
3. Key Sanskrit terms with brief explanations
4. Relevance to the question (3-5 sentences total)
"""
    else:
        special_instruction = "Be concise (3-5 sentences) yet substantive."

    return (
        f"{base_role}\n\n"
        f"{special_instruction}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )

# ---------------------------------------------------------------------------
# LLM call（Gemini）
# ---------------------------------------------------------------------------

def ask_gemini(prompt: str, retries: int = 3, delay: float = 5.0) -> str:
    generation_config = genai.types.GenerationConfig(
        temperature=0.2,
        max_output_tokens=512,
    )
    for attempt in range(retries):
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            return response.text.strip()
        except Exception as exc:
            log.warning("Gemini error (attempt %d/%d): %s", attempt + 1, retries, exc)
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    return "ERROR: Could not generate answer."

# ---------------------------------------------------------------------------
# 增强版 answer_question（整合 robust_retrieve + consciousness prompt）
# ---------------------------------------------------------------------------

def answer_question_enhanced(
    vectorstore: FAISS,
    question: str,
    round_num: int,
    translations: List[str],
) -> Tuple[str, List[str], dict]:
    """
    主要入口：使用 llm2_updated.py 的增强检索 + consciousness 专项 prompt。
    返回：(answer, sources, metadata)
    metadata 记录检索细节，供论文分析使用。
    """
    start = time.time()

    # 增强检索
    docs = robust_retrieve(vectorstore, question, k=TOP_K, mmr_lambda=MMR_LAMBDA)

    retrieval_time = time.time() - start

    if not docs:
        return "I don't know.", [], {"retrieval_time": retrieval_time, "docs_retrieved": 0}

    # 构建 prompt
    prompt = build_prompt(question, docs, round_num, translations)

    # 生成答案
    gen_start = time.time()
    answer = ask_gemini(prompt)
    generation_time = time.time() - gen_start

    sources = list({d.metadata.get("source", "unknown") for d in docs})

    metadata = {
        "retrieval_time":    round(retrieval_time, 3),
        "generation_time":   round(generation_time, 3),
        "total_time":        round(time.time() - start, 3),
        "docs_retrieved":    len(docs),
        "is_consciousness":  is_consciousness_query(question),
        "query_expansions":  expand_query(question),
        "context_tokens":    count_tokens("\n\n".join(d.page_content for d in docs)),
    }

    return answer, sources, metadata

# ---------------------------------------------------------------------------
# Text loading（与原脚本一致，chunk size 对齐 llm2_updated.py）
# ---------------------------------------------------------------------------

def load_translation(name: str, filename: str) -> List[Document]:
    path = GITA_DIR / filename
    if not path.exists():
        log.warning("File not found, skipping: %s", path)
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    log.info("Loaded %s — %d chars", name, len(text))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [
        Document(page_content=chunk, metadata={"source": name, "chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]

# ---------------------------------------------------------------------------
# FAISS 索引管理（增量构建，与原脚本一致）
# ---------------------------------------------------------------------------

def build_or_load_index(
    round_num: int,
    new_docs: List[Document],
    prev_round_num: Optional[int] = None,
) -> FAISS:
    index_path = INDEX_BASE / f"round_{round_num:02d}"
    if index_path.exists():
        log.info("Loading cached index for round %d", round_num)
        return FAISS.load_local(
            str(index_path), embeddings, allow_dangerous_deserialization=True
        )

    prev_path = INDEX_BASE / f"round_{prev_round_num:02d}" if prev_round_num else None

    if prev_path and prev_path.exists():
        log.info("Extending R%d index with %d new chunks…", prev_round_num, len(new_docs))
        vectorstore = FAISS.load_local(
            str(prev_path), embeddings, allow_dangerous_deserialization=True
        )
        texts = [doc.page_content for doc in new_docs]
        metadatas = [doc.metadata for doc in new_docs]
        new_vecs = embeddings.embed_documents(texts)
        vectorstore.add_embeddings(list(zip(texts, new_vecs)), metadatas=metadatas)
    else:
        log.info("Building R%d index from scratch (%d chunks)…", round_num, len(new_docs))
        vectorstore = FAISS.from_documents(new_docs, embeddings)

    vectorstore.save_local(str(index_path))
    return vectorstore

# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def run_rounds(target_rounds: Optional[List[int]] = None) -> List[dict]:
    existing = []
    if QA_JSON.exists():
        try:
            existing = json.loads(QA_JSON.read_text(encoding="utf-8"))
            log.info("Resuming — found %d existing records", len(existing))
        except (json.JSONDecodeError, ValueError):
            log.warning("qa_results_enhanced.json is empty or corrupt, starting fresh.")
            existing = []

    done_keys = {(r["round_number"], r["question"]) for r in existing}
    results = list(existing)
    cumulative_names: List[str] = []
    last_built_round: Optional[int] = None

    for round_idx, (trans_name, trans_file) in enumerate(TRANSLATIONS, start=1):
        new_docs = load_translation(trans_name, trans_file)

        if target_rounds and round_idx not in target_rounds:
            cumulative_names.append(trans_name)
            if new_docs:
                build_or_load_index(round_idx, new_docs, prev_round_num=last_built_round)
                last_built_round = round_idx
            continue

        if not new_docs:
            log.warning("Round %d: no docs for %s, skipping.", round_idx, trans_name)
            cumulative_names.append(trans_name)
            continue

        cumulative_names.append(trans_name)
        log.info("=== Round %d | %d translation(s) | %d new chunks ===",
                 round_idx, len(cumulative_names), len(new_docs))

        vectorstore = build_or_load_index(round_idx, new_docs, prev_round_num=last_built_round)
        last_built_round = round_idx

        for q in QUESTIONS:
            if (round_idx, q) in done_keys:
                log.info("  [cached] R%d Q: %s", round_idx, q[:50])
                continue

            log.info("  Answering: %s", q[:60])

            # 使用增强版检索
            answer, sources, meta = answer_question_enhanced(
                vectorstore, q, round_idx, list(cumulative_names)
            )

            if answer.startswith("ERROR"):
                log.warning("  [skip cache] failed: %s", q[:50])
                continue

            record = {
                "round_number":          round_idx,
                "translations_included": list(cumulative_names),
                "question":              q,
                "answer":                answer,
                "retrieved_sources":     sources,
                "timestamp":             datetime.now().isoformat(),
                # 新增：记录增强检索的元数据（供论文分析）
                "retrieval_metadata":    meta,
            }
            results.append(record)
            done_keys.add((round_idx, q))

            QA_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            time.sleep(1.0)

    df = pd.DataFrame(results)
    df.to_csv(QA_CSV, index=False)
    log.info("QA results → %s", QA_CSV)
    return results

# ---------------------------------------------------------------------------
# 评估（BERTScore + N-gram，与原脚本一致）
# ---------------------------------------------------------------------------

STOP_WORDS = set(stopwords.words("english"))

def extract_ngrams(text: str, n: int, top_k: int = 20) -> List[Tuple[str, int]]:
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in STOP_WORDS]
    grams = ngrams(tokens, n)
    return Counter(grams).most_common(top_k)

def compute_bertscore(results: List[dict]) -> pd.DataFrame:
    from bert_score import score as bert_score_fn
    df = pd.DataFrame(results)
    rows = []
    for question in QUESTIONS:
        q_df = df[df["question"] == question].sort_values("round_number").reset_index(drop=True)
        if len(q_df) < 2:
            continue
        answers = q_df["answer"].tolist()
        rounds  = q_df["round_number"].tolist()
        for i in range(len(answers) - 1):
            try:
                P, R, F1 = bert_score_fn([answers[i+1]], [answers[i]], lang="en", verbose=False)
                rows.append({
                    "question":            question,
                    "is_consciousness":    is_consciousness_query(question),
                    "round_from":          rounds[i],
                    "round_to":            rounds[i+1],
                    "bertscore_precision": round(float(P[0]), 4),
                    "bertscore_recall":    round(float(R[0]), 4),
                    "bertscore_f1":        round(float(F1[0]), 4),
                })
            except Exception as exc:
                log.warning("BERTScore failed: %s", exc)

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(EVAL_CSV, index=False)
    log.info("BERTScore → %s", EVAL_CSV)
    return eval_df

def compute_ngram_analysis(results: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    rows = []
    for round_num in sorted(df["round_number"].unique()):
        r_df = df[df["round_number"] == round_num]
        all_text = " ".join(r_df["answer"].tolist())

        # 单独分析 consciousness 问题的 n-gram
        c_df = r_df[r_df["question"].apply(is_consciousness_query)]
        c_text = " ".join(c_df["answer"].tolist()) if not c_df.empty else ""

        translations = r_df["translations_included"].iloc[0]
        for n, label in [(2, "bigram"), (3, "trigram")]:
            for gram, count in extract_ngrams(all_text, n, top_k=15):
                rows.append({
                    "round_number":          round_num,
                    "translations_included": ", ".join(translations),
                    "ngram_type":            label,
                    "ngram":                 " ".join(gram),
                    "count":                 count,
                    "subset":                "all",
                })
            # consciousness 子集
            if c_text:
                for gram, count in extract_ngrams(c_text, n, top_k=10):
                    rows.append({
                        "round_number":          round_num,
                        "translations_included": ", ".join(translations),
                        "ngram_type":            label,
                        "ngram":                 " ".join(gram),
                        "count":                 count,
                        "subset":                "consciousness_only",
                    })

    ngram_df = pd.DataFrame(rows)
    ngram_df.to_csv(NGRAM_CSV, index=False)
    log.info("N-gram → %s", NGRAM_CSV)
    return ngram_df

def generate_comparison_table(results: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    rounds = sorted(df["round_number"].unique())

    bs_lookup: dict = {}
    try:
        eval_df = pd.read_csv(EVAL_CSV)
        if not eval_df.empty:
            for _, row in eval_df.iterrows():
                bs_lookup[(int(row["round_to"]), row["question"])] = row["bertscore_f1"]
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pass

    rows = []
    for question in QUESTIONS:
        q_df = df[df["question"] == question].sort_values("round_number")
        row: dict = {
            "question": question,
            "is_consciousness": is_consciousness_query(question),
        }
        for rn in rounds:
            match = q_df[q_df["round_number"] == rn]
            if match.empty:
                row[f"R{rn}_answer"] = ""
                row[f"R{rn}_bertscore_f1"] = ""
                continue
            answer = match["answer"].iloc[0]
            snippet = answer[:200].replace("\n", " ") + ("…" if len(answer) > 200 else "")
            row[f"R{rn}_answer"] = snippet
            row[f"R{rn}_bertscore_f1"] = bs_lookup.get((rn, question), "")
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(COMPARISON_CSV, index=False)
    log.info("Comparison table → %s", COMPARISON_CSV)
    return comp_df

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: List[dict], eval_df: pd.DataFrame, ngram_df: pd.DataFrame):
    df = pd.DataFrame(results)
    rounds = sorted(df["round_number"].unique())

    print("\n" + "=" * 70)
    print("BHAGAVAD GITA RAG EVALUATION (Gemini) — SUMMARY")
    print("=" * 70)
    print(f"Total rounds:    {len(rounds)}")
    print(f"Total Q&A pairs: {len(df)}")

    # Consciousness 问题单独统计
    c_df = df[df["question"].apply(is_consciousness_query)]
    print(f"Consciousness Q&A pairs: {len(c_df)} / {len(df)}")

    if not eval_df.empty:
        print("\n--- BERTScore F1 (mean per round transition) ---")
        summary = eval_df.groupby(["round_from", "round_to"])["bertscore_f1"].mean().reset_index()
        for _, row in summary.iterrows():
            print(f"  R{int(row.round_from):2d} → R{int(row.round_to):2d} : F1 = {row.bertscore_f1:.4f}")

        # Consciousness 问题的 BERTScore 单独输出
        if "is_consciousness" in eval_df.columns:
            c_eval = eval_df[eval_df["is_consciousness"] == True]
            if not c_eval.empty:
                print(f"\n--- BERTScore F1 (consciousness questions only) ---")
                c_summary = c_eval.groupby(["round_from", "round_to"])["bertscore_f1"].mean().reset_index()
                for _, row in c_summary.iterrows():
                    print(f"  R{int(row.round_from):2d} → R{int(row.round_to):2d} : F1 = {row.bertscore_f1:.4f}")

    print("\n--- Output files ---")
    for p in [QA_JSON, QA_CSV, EVAL_CSV, NGRAM_CSV, COMPARISON_CSV]:
        print(f"  {p}")
    print("=" * 70)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", nargs="+", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-bertscore", action="store_true")
    args = parser.parse_args()

    if not args.eval_only:
        results = run_rounds(target_rounds=args.rounds)
    else:
        if not QA_JSON.exists():
            sys.exit("No existing results. Run without --eval-only first.")
        try:
            results = json.loads(QA_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            sys.exit("qa_results_enhanced.json is empty or corrupt. Delete it and re-run without --eval-only.")
        log.info("Loaded %d records for evaluation.", len(results))

    eval_df  = pd.DataFrame()
    ngram_df = pd.DataFrame()

    if not args.no_bertscore:
        log.info("Computing BERTScore…")
        eval_df = compute_bertscore(results)

    log.info("Computing N-gram analysis…")
    ngram_df = compute_ngram_analysis(results)

    log.info("Generating comparison table…")
    generate_comparison_table(results)

    print_summary(results, eval_df, ngram_df)


if __name__ == "__main__":
    main()