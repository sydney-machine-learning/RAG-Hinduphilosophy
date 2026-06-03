"""
Bhagavad Gita RAG Evaluation Pipeline
======================================
Incrementally loads translations, runs 10 fixed questions per round,
records answers, then evaluates semantic drift using BERTScore,
bigrams/trigrams, and a cross-round comparison table.

Usage:
    python gita_rag_eval.py                  # full run
    python gita_rag_eval.py --rounds 1 2 3   # specific rounds only
    python gita_rag_eval.py --eval-only      # skip QA, only run evaluation
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
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# NLTK SSL fix for macOS
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

# Ordered list of translations for incremental loading.
# Preprocessed (_clean) versions are used where available.
TRANSLATIONS = [
    ("Gita-Swarupananda",         "Gita-Swarupananda.txt"),
    ("Radhakrishnan",              "Bhagavad_Gita_Radhakrishnan.txt"),
    ("Prabhupada",                 "The Bhagavad Gita - Prabhupada.txt"),
    ("Eknath-Easwaran",            "The_bhagavad_gita_eknath_easwaran.txt"),
    ("Mahesh-Yogi",                "bhagavad-gita-by-mahesh-yogi_clean.txt"),
    ("Chinmayananda",              "Holy Geeta by Swami Chinmayananda .txt"),
    ("Vinoba-Bhave",               "Talks on the Gita by Vinoba Bhave.txt"),
    ("Dayananda-Intro",            "Introduction_to_Gita_Dayananda.txt"),
    ("Gita-Press-Roman",           "455_gita_roman_clean.txt"),
    ("Dnyaneshwari",               "Dnyanehwari.txt"),
    ("Gita-Article",               "Gita_article_published_version-libre.txt"),
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

RESULTS_DIR    = Path("gita_eval_results")
QA_JSON        = RESULTS_DIR / "qa_results.json"
QA_CSV         = RESULTS_DIR / "qa_results.csv"
EVAL_CSV       = RESULTS_DIR / "evaluation_bertscore.csv"
NGRAM_CSV      = RESULTS_DIR / "ngram_analysis.csv"
COMPARISON_CSV = RESULTS_DIR / "comparison_table.csv"
INDEX_BASE     = RESULTS_DIR / "faiss_indices"

# RAG parameters
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
TOP_K         = 5

# DeepSeek model
LLM_MODEL = "deepseek-chat"

RESULTS_DIR.mkdir(exist_ok=True)
INDEX_BASE.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# LangChain / DeepSeek imports
# ---------------------------------------------------------------------------

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    sys.exit("ERROR: Set DEEPSEEK_API_KEY in your .env file.")
deepseek_client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# ---------------------------------------------------------------------------
# Embedding model — local sentence-transformers (no API quota, no rate limits)
# Gemini API is used ONLY for LLM answer generation, not for embedding.
# ---------------------------------------------------------------------------
log.info("Loading local embedding model (sentence-transformers)…")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# ---------------------------------------------------------------------------
# Text loading & chunking
# ---------------------------------------------------------------------------

def load_translation(name: str, filename: str) -> list[Document]:
    path = GITA_DIR / filename
    if not path.exists():
        log.warning("File not found, skipping: %s", path)
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    # Basic cleanup: collapse excessive whitespace
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
# FAISS index management
# ---------------------------------------------------------------------------

def build_or_load_index(
    round_num: int,
    new_docs: list[Document],
    prev_round_num: int | None = None,
) -> FAISS:
    """
    Load cached FAISS index for this round, or build it efficiently:
      - Round 1: embed new_docs from scratch.
      - Round N (N>1): load round N-1 index and add only new_docs.
    This avoids re-embedding documents from previous rounds on every run.
    """
    index_path = INDEX_BASE / f"round_{round_num:02d}"
    if index_path.exists():
        log.info("Loading cached index for round %d", round_num)
        return FAISS.load_local(
            str(index_path), embeddings, allow_dangerous_deserialization=True
        )

    prev_path = INDEX_BASE / f"round_{prev_round_num:02d}" if prev_round_num else None

    if prev_path and prev_path.exists():
        log.info(
            "Building round %d index: loading R%d base + embedding %d new chunks…",
            round_num, prev_round_num, len(new_docs),
        )
        vectorstore = FAISS.load_local(
            str(prev_path), embeddings, allow_dangerous_deserialization=True
        )
        # Use add_embeddings (not add_documents) so embedding goes through
        # embeddings.embed_documents — avoiding FAISS's internal one-by-one loop.
        texts = [doc.page_content for doc in new_docs]
        metadatas = [doc.metadata for doc in new_docs]
        log.info("  Embedding %d new chunks…", len(texts))
        new_vecs = embeddings.embed_documents(texts)
        vectorstore.add_embeddings(list(zip(texts, new_vecs)), metadatas=metadatas)
    else:
        log.info(
            "Building FAISS index for round %d from scratch (%d chunks)…",
            round_num, len(new_docs),
        )
        vectorstore = FAISS.from_documents(new_docs, embeddings)

    vectorstore.save_local(str(index_path))
    return vectorstore

# ---------------------------------------------------------------------------
# RAG answer generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a knowledgeable scholar of the Bhagavad Gita and Hindu philosophy. "
    "Answer the question using ONLY the provided context passages. "
    "Be concise (3-5 sentences) yet substantive. "
    "If the context does not contain enough information, say so briefly."
)

def build_prompt(question: str, context_docs: list[Document]) -> str:
    context_text = "\n\n---\n\n".join(
        f"[Source: {d.metadata['source']}]\n{d.page_content}"
        for d in context_docs
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )


def ask_deepseek(prompt: str, retries: int = 3, delay: float = 5.0) -> str:
    for attempt in range(retries):
        try:
            response = deepseek_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            log.warning("DeepSeek error (attempt %d/%d): %s", attempt + 1, retries, exc)
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    return "ERROR: Could not generate answer."


def answer_question(vectorstore: FAISS, question: str) -> tuple[str, list[str]]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    context_docs = retriever.invoke(question)
    prompt = build_prompt(question, context_docs)
    answer = ask_deepseek(prompt)
    sources = list({d.metadata["source"] for d in context_docs})
    return answer, sources

# ---------------------------------------------------------------------------
# Round runner
# ---------------------------------------------------------------------------

def run_rounds(target_rounds: Optional[list[int]] = None) -> list[dict]:
    """Run QA for each round. Returns list of result records."""
    # Load existing results to allow resuming
    existing = []
    if QA_JSON.exists():
        existing = json.loads(QA_JSON.read_text())
        log.info("Resuming — found %d existing records", len(existing))

    done_keys = {(r["round_number"], r["question"]) for r in existing}
    results = list(existing)

    cumulative_names: list[str] = []
    last_built_round: int | None = None  # tracks the most recent successfully built index

    for round_idx, (trans_name, trans_file) in enumerate(TRANSLATIONS, start=1):
        new_docs = load_translation(trans_name, trans_file)

        if target_rounds and round_idx not in target_rounds:
            # Skip QA for this round, but we still need its index for future rounds
            cumulative_names.append(trans_name)
            if new_docs:
                # Silently build/cache the index so later rounds can extend it
                build_or_load_index(round_idx, new_docs, prev_round_num=last_built_round)
                last_built_round = round_idx
            continue

        if not new_docs:
            log.warning("Round %d: no documents loaded for %s, skipping.", round_idx, trans_name)
            cumulative_names.append(trans_name)
            continue

        cumulative_names.append(trans_name)

        log.info(
            "=== Round %d | %d translation(s) | adding %d new chunks ===",
            round_idx, len(cumulative_names), len(new_docs),
        )

        vectorstore = build_or_load_index(round_idx, new_docs, prev_round_num=last_built_round)
        last_built_round = round_idx

        for q in QUESTIONS:
            if (round_idx, q) in done_keys:
                log.info("  [cached] R%d Q: %s", round_idx, q[:50])
                continue

            log.info("  Answering: %s", q[:60])
            answer, sources = answer_question(vectorstore, q)

            # 只缓存成功的答案，ERROR 答案不写入 done_keys，下次会重试
            if answer.startswith("ERROR"):
                log.warning("  [skip cache] answer failed, will retry next run: %s", q[:50])
                continue

            record = {
                "round_number": round_idx,
                "translations_included": list(cumulative_names),
                "question": q,
                "answer": answer,
                "retrieved_sources": sources,
                "timestamp": datetime.now().isoformat(),
            }
            results.append(record)
            done_keys.add((round_idx, q))

            # Save incrementally after each answer
            QA_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            time.sleep(1.0)  # stay within Gemini free-tier rate limits

    # Also write CSV
    df = pd.DataFrame(results)
    df.to_csv(QA_CSV, index=False)
    log.info("QA results saved → %s and %s", QA_JSON, QA_CSV)
    return results

# ---------------------------------------------------------------------------
# Evaluation — BERTScore
# ---------------------------------------------------------------------------

def compute_bertscore(results: list[dict]) -> pd.DataFrame:
    """
    For each question, compare answer from round N vs round N-1
    using BERTScore (semantic similarity).
    """
    from bert_score import score as bert_score_fn

    df = pd.DataFrame(results)
    rows = []

    for question in QUESTIONS:
        q_df = df[df["question"] == question].sort_values("round_number").reset_index(drop=True)
        if len(q_df) < 2:
            continue

        answers = q_df["answer"].tolist()
        rounds  = q_df["round_number"].tolist()

        # Compare each consecutive pair
        for i in range(len(answers) - 1):
            ref = [answers[i]]
            cand = [answers[i + 1]]
            try:
                P, R, F1 = bert_score_fn(cand, ref, lang="en", verbose=False)
                rows.append({
                    "question":         question,
                    "round_from":       rounds[i],
                    "round_to":         rounds[i + 1],
                    "bertscore_precision": round(float(P[0]), 4),
                    "bertscore_recall":    round(float(R[0]), 4),
                    "bertscore_f1":        round(float(F1[0]), 4),
                })
            except Exception as exc:
                log.warning("BERTScore failed for Q%d: %s", i, exc)

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(EVAL_CSV, index=False)
    log.info("BERTScore results saved → %s", EVAL_CSV)
    return eval_df

# ---------------------------------------------------------------------------
# Evaluation — N-gram analysis
# ---------------------------------------------------------------------------

STOP_WORDS = set(stopwords.words("english"))


def extract_ngrams(text: str, n: int, top_k: int = 20) -> list[tuple[str, int]]:
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in STOP_WORDS]
    grams = ngrams(tokens, n)
    counts = Counter(grams)
    return counts.most_common(top_k)


def compute_ngram_analysis(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    rows = []

    for round_num in sorted(df["round_number"].unique()):
        r_df = df[df["round_number"] == round_num]
        all_text = " ".join(r_df["answer"].tolist())
        translations = r_df["translations_included"].iloc[0]

        for n, label in [(2, "bigram"), (3, "trigram")]:
            top = extract_ngrams(all_text, n, top_k=15)
            for gram, count in top:
                rows.append({
                    "round_number":          round_num,
                    "translations_included": ", ".join(translations),
                    "ngram_type":            label,
                    "ngram":                 " ".join(gram),
                    "count":                 count,
                })

    ngram_df = pd.DataFrame(rows)
    ngram_df.to_csv(NGRAM_CSV, index=False)
    log.info("N-gram analysis saved → %s", NGRAM_CSV)
    return ngram_df

# ---------------------------------------------------------------------------
# Evaluation — Comparison table
# ---------------------------------------------------------------------------

def generate_comparison_table(results: list[dict]) -> pd.DataFrame:
    """
    Pivot table: rows = questions, columns = round numbers,
    cells = truncated answer + BERTScore F1 vs previous round.
    """
    df = pd.DataFrame(results)
    rounds = sorted(df["round_number"].unique())

    # Compute BERTScore lookup (gracefully handle missing or empty file)
    bs_lookup: dict[tuple[int, str], float] = {}
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
        row: dict = {"question": question}

        for rn in rounds:
            match = q_df[q_df["round_number"] == rn]
            if match.empty:
                row[f"R{rn}_answer"] = ""
                row[f"R{rn}_bertscore_f1"] = ""
                continue
            answer = match["answer"].iloc[0]
            # Truncate for readability in table
            snippet = answer[:200].replace("\n", " ") + ("…" if len(answer) > 200 else "")
            row[f"R{rn}_answer"] = snippet
            bs = bs_lookup.get((rn, question), "")
            row[f"R{rn}_bertscore_f1"] = bs

        rows.append(row)

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(COMPARISON_CSV, index=False)
    log.info("Comparison table saved → %s", COMPARISON_CSV)
    return comp_df

# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], eval_df: pd.DataFrame, ngram_df: pd.DataFrame):
    df = pd.DataFrame(results)
    rounds = sorted(df["round_number"].unique())

    print("\n" + "=" * 70)
    print("BHAGAVAD GITA RAG EVALUATION — SUMMARY")
    print("=" * 70)
    print(f"Total rounds:    {len(rounds)}")
    print(f"Total Q&A pairs: {len(df)}")
    print(f"Questions/round: {len(QUESTIONS)}")

    if not eval_df.empty:
        print("\n--- BERTScore F1 (mean per round transition) ---")
        summary = (
            eval_df.groupby(["round_from", "round_to"])["bertscore_f1"]
            .mean()
            .reset_index()
        )
        for _, row in summary.iterrows():
            print(
                f"  R{int(row.round_from):2d} → R{int(row.round_to):2d} : "
                f"mean F1 = {row.bertscore_f1:.4f}"
            )

    if not ngram_df.empty:
        print("\n--- Top trigrams in final round ---")
        last_round = ngram_df["round_number"].max()
        top_trigrams = (
            ngram_df[
                (ngram_df["round_number"] == last_round)
                & (ngram_df["ngram_type"] == "trigram")
            ]
            .head(10)
        )
        for _, row in top_trigrams.iterrows():
            print(f"  {row['ngram']!r:35s}  count={row['count']}")

    print("\n--- Output files ---")
    for p in [QA_JSON, QA_CSV, EVAL_CSV, NGRAM_CSV, COMPARISON_CSV]:
        print(f"  {p}")
    print("=" * 70)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bhagavad Gita RAG Evaluation Pipeline")
    parser.add_argument(
        "--rounds", nargs="+", type=int, default=None,
        help="Run only specific round numbers (e.g. --rounds 1 3 5)",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip QA generation; only run evaluation on existing results",
    )
    parser.add_argument(
        "--no-bertscore", action="store_true",
        help="Skip BERTScore computation (faster, for testing)",
    )
    args = parser.parse_args()

    # --- Step 1: QA generation ---
    if not args.eval_only:
        results = run_rounds(target_rounds=args.rounds)
    else:
        if not QA_JSON.exists():
            sys.exit("No existing results found. Run without --eval-only first.")
        results = json.loads(QA_JSON.read_text())
        log.info("Loaded %d existing records for evaluation.", len(results))

    # --- Step 2: Evaluation ---
    eval_df   = pd.DataFrame()
    ngram_df  = pd.DataFrame()

    if not args.no_bertscore:
        log.info("Computing BERTScore…")
        eval_df = compute_bertscore(results)

    log.info("Computing N-gram analysis…")
    ngram_df = compute_ngram_analysis(results)

    log.info("Generating comparison table…")
    generate_comparison_table(results)

    # --- Step 3: Summary ---
    print_summary(results, eval_df, ngram_df)


if __name__ == "__main__":
    main()
