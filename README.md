# TechNova Policy Q&A

A small Retrieval-Augmented Generation (RAG) project built for the **Introduction to LangChain: Chains, Retrieval and RAG** workshop.

The program answers questions about the employee policies of a fictional company, TechNova Solutions. It retrieves relevant sections of the policy document and passes them to an LLM to generate an answer.

## How it works

1. Loads the supplied policy text using LangChain's `TextLoader`.
2. Splits the text into chunks using `RecursiveCharacterTextSplitter`.
3. Creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
4. Stores the embeddings in FAISS.
5. Retrieves the three most relevant chunks for each question.
6. Uses a prompt template and a Groq-hosted LLM to answer using the retrieved context.

The policy text is included in the Python file, so no separate document download is needed. A chunk size of 350 characters keeps individual policies and their conditions together.

## Tools used

- Python
- LangChain
- Hugging Face sentence-transformers
- FAISS
- Groq (`openai/gpt-oss-120b`)

## Run in Google Colab

Create a notebook at [Google Colab](https://colab.research.google.com/) and run the following cells in order.

### 1. Install dependencies

```python
!pip install -q langchain-core langchain-community langchain-text-splitters langchain-huggingface langchain-groq sentence-transformers faiss-cpu
```

If Colab reports that it requires `requests==2.32.4`, run:

```python
!pip install -q requests==2.32.4
```

### 2. Upload the Python file

```python
from google.colab import files
files.upload()
```

Select `25f2007217.py` from this repository.

### 3. Run the program

```python
%run /content/25f2007217.py
```

Enter your own [Groq API key](https://console.groq.com/keys) at the hidden prompt. The program does not save the key in the source file.

The first run downloads the embedding model. Internet access is required for that download and for Groq requests.

### 4. Download the file with its outputs

After all three questions finish, the program saves the generated answers as comments at the end of the Python file.

```python
files.download("/content/25f2007217.py")
```

## Run locally

Install the dependencies in a Python environment:

```bash
python -m pip install langchain-core langchain-community langchain-text-splitters langchain-huggingface langchain-groq sentence-transformers faiss-cpu
python 25f2007217.py
```

Use `%run` only in Colab or IPython, not in a PowerShell terminal. If you rename the Python file, update the commands to match.

## Test questions

1. How many paid annual leave days does a full-time employee receive per year?
2. An employee wants to work remotely on Wednesday and take leave on Friday. What do the company policies say about these two requests?
3. What is the company's parental leave policy?

### Expected behavior

| Question | What the answer should cover |
| --- | --- |
| Annual leave | Full-time employees receive 18 paid annual leave days per calendar year. |
| Wednesday remote work and Friday leave | Remote work on Wednesday requires an email at least one working day in advance and manager approval. Friday annual leave requires a request at least three calendar days in advance and manager confirmation. |
| Parental leave | The provided document does not contain this information. |

These are expected results, not a transcript of a particular run. Actual generated outputs are recorded at the end of the Python file after execution.

## Notes

- Never commit API keys or other credentials to GitHub.
- This is a learning project using fictional policies, not a production HR system.
- The prompt instructs the model to avoid unsupported answers, but generated answers should still be checked against the document.
- The script updates its own output comments after a successful run, so it needs write access to its file.
