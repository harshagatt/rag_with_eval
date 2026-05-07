# evaluate_rag.py
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient

from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig
from ragas.metrics import (
    Faithfulness,
    # ResponseRelevancy is intentionally excluded:
    # It asks the LLM to output strict JSON {question, noncommittal} with no
    # preamble. Local models like llama3.1 always add explanation text before
    # the JSON, causing OutputParserException on every job. This is a prompt
    # compliance issue — not fixable with timeouts or retries.
    LLMContextPrecisionWithoutReference,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

import json, os, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
You are a financial analyst assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(collection_name: str):
    """Builds the RAG pipeline using LCEL."""
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # RAG chain uses llama3.1 without format="json" — free-text answer is fine here
    llm = ChatOllama(model="llama3.1", temperature=0)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever


def build_ragas_config():
    """
    Configures Ragas to use local llama3.1 as its judge.

    format="json" is set on the Ragas LLM only — this tells Ollama to
    enforce strict JSON output for every Ragas scoring prompt, preventing
    the model from adding explanation text that breaks the output parser.

    RunConfig:
      max_workers=1  → sequential; Ollama handles one request at a time
      timeout=300    → 5 min per call; local inference is slow
      max_retries=2  → retry twice before marking a sample as nan
    """
    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # format="json" is the key fix — forces Ollama to output raw JSON only,
    # no preamble text, matching what Ragas's output parser expects
    ragas_llm = LangchainLLMWrapper(
        ChatOllama(model="llama3.1", temperature=0, base_url=ollama_base, format="json")
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base)
    )
    run_config = RunConfig(
        timeout=300,
        max_retries=2,
        max_workers=1,
    )
    return ragas_llm, ragas_embeddings, run_config


def run_evaluation(ground_truth_path: str, collection_name: str):
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)

    print("Building RAG pipeline...")
    rag_chain, retriever = build_rag_chain(collection_name)

    samples = []
    for item in ground_truth:
        question = item["question"]
        print(f"\nQ: {question}")

        answer = rag_chain.invoke(question)
        source_docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in source_docs]
        print(f"A: {answer[:200]}...")

        samples.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        ))

    dataset = EvaluationDataset(samples=samples)

    print("\nConfiguring Ragas to use local llama3.1 with JSON mode...")
    ragas_llm, ragas_embeddings, run_config = build_ragas_config()

    # Only metrics whose Ragas prompts work reliably with local LLMs:
    #   Faithfulness                    → verifies each claim against retrieved chunks
    #   LLMContextPrecisionWithoutRef   → checks if retrieved chunks were actually useful
    # ResponseRelevancy is excluded — its prompt requires strict {question, noncommittal}
    # JSON that local models cannot reliably produce without adding preamble text.
    metrics = [
        Faithfulness(),
        LLMContextPrecisionWithoutReference(),
    ]

    print(f"Running {len(metrics)} metrics on {len(samples)} samples (sequential, ~{len(samples) * 2} mins)...")
    score = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    df = score.to_pandas()

    print("\n" + "=" * 55)
    print("RAGAS EVALUATION RESULTS  [llama3.1 local judge]")
    print("=" * 55)

    metric_display = [
        ("faithfulness",                           "Faithfulness                    ", 0.80),
        ("llm_context_precision_without_reference","Context Precision (no reference)", 0.70),
    ]
    for col, label, target in metric_display:
        if col in df.columns:
            val = df[col].mean()
            flag = "✓" if val >= target else "✗ needs work"
            print(f"  {label}: {val:.3f}  (target > {target})  {flag}")
        else:
            print(f"  {label}: not computed — check warnings above")

    print("=" * 55)
    print("\nWhat each score means:")
    print("  Faithfulness:      does the answer use ONLY the retrieved chunks? (no hallucination)")
    print("  Context Precision: were the retrieved chunks actually relevant to the question?")
    print("\nScore scale:  1.0 = perfect | 0.8+ = good | 0.6-0.8 = needs work | <0.6 = broken")

    return score


if __name__ == "__main__":
    scores = run_evaluation(
        ground_truth_path="ground_truth.json",
        collection_name=os.getenv("COLLECTION_NAME", "financial_docs")
    )