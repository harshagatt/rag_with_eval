RAG Evaluation Pipeline: Apple Financial Analysis
This project provides a robust evaluation framework for a Retrieval-Augmented Generation (RAG) pipeline using Ragas and Ollama. It is specifically configured to analyze complex financial documents, such as the Apple Inc. FY25 Q2 Consolidated Financial Statements.

The pipeline utilizes local models to ensure data privacy and eliminates reliance on paid API services like OpenAI.

🚀 Key Features

Local-First Evaluation: Uses llama3.1 via Ollama as the "Judge" LLM to score RAG performance.

Financial Domain Focus: Tailored to evaluate extraction accuracy from dense financial tables, including Net Sales, Operating Income, and Cash Flow metrics.

Ragas 0.2+ Integration: Implements the latest SingleTurnSample and EvaluationDataset schemas.

Stability Enhancements: Configured with extended timeouts and concurrency limits to prevent TimeoutError on local hardware.

Debug-Ready: Automatically exports evaluation results to evaluation_debug.csv for row-by-row analysis.

🛠️ Technical Stack

LLM (RAG & Judge): Ollama (llama3.1)

Embeddings: Ollama (nomic-embed-text)

Vector Database: Qdrant

Framework: LangChain & Ragas

Data Source: Apple Inc. Condensed Consolidated Statements (Unaudited) - You can choose any files with lots of tables so that you can
test the effectiveness of the PDF parsing

📋 Prerequisites (MacOS / Homebrew)
Install Ollama:

Bash
brew install ollama
Pull Required Models:

Bash
ollama pull llama3.1
ollama pull nomic-embed-text
Install Python Dependencies:

Bash
pip install ragas langchain_ollama langchain_qdrant qdrant-client pandas numpy python-dotenv
⚙️ Configuration
Create a .env file in the root directory:

Code snippet
OLLAMA_BASE_URL=http://localhost:11434
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=financial_docs
📊 Evaluation Metrics
The pipeline measures performance across three critical dimensions:

Faithfulness: Ensures the answer is derived strictly from the provided financial context (e.g., verifying if the reported Net Income of $61,110 million matches the document).

Answer Relevancy: Assesses how well the response addresses the specific financial query.

Context Precision: Evaluates the retriever's ability to rank the correct financial tables (e.g., the Balance Sheet vs. Cash Flow Statement) at the top.

📝 Usage
To run the evaluation:

Bash
python evaluate_rag.py
Upon completion, the script will output a summary to the terminal and generate evaluation_debug.csv. If any metric returns NaN, check the debug file to verify if the Judge LLM encountered a parsing error or a timeout.

⚠️ Troubleshooting Local Timeouts
If you encounter TimeoutError() during evaluation:
Increase Timeout: The build_ragas_config() is set to 300 seconds to accommodate complex table reasoning.

Reduce Workers: Concurrency is limited via run_config={"max_workers": 2} to prevent CPU/GPU exhaustion on local Mac hardware.
Increase Timeout: The build_ragas_config() is set to 300 seconds to accommodate complex table reasoning.

Reduce Workers: Concurrency is limited via run_config={"max_workers": 2} to prevent CPU/GPU exhaustion on local Mac hardware.
