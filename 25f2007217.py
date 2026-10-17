# Email: 25f2007217@ds.study.iitm.ac.in
# 25f2007217
# Introduction to LangChain: Chains, Retrieval and RAG

from getpass import getpass
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Policy text copied from the document provided in the assignment.
policy_text = """TechNova Solutions - Employee Policy Document

1. Annual Leave: Every full-time employee receives 18 paid annual leave days per calendar year. Employees must submit annual leave requests to their manager at least 3 calendar days before the requested leave date. Leave approval depends on the manager's confirmation.

2. Work From Home: Employees may work remotely on Wednesdays and Fridays. Employees must inform their manager by email at least one working day in advance. Remote work is subject to manager approval.

3. Working Hours: The standard working day is from 9:30 AM to 5:30 PM, Monday through Friday. Employees receive a one-hour lunch break.

4. Sick Leave: Employees receive 10 paid sick leave days per calendar year. If an employee is absent due to illness for more than two consecutive working days, a medical certificate must be submitted to HR.

5. Learning Allowance: Each full-time employee can claim up to ₹12,000 per calendar year for approved job-related courses and certifications. Claims must include proof of payment and course completion.
"""


def main():
    api_key = getpass("Enter your Groq API key: ").strip()
    if not api_key:
        raise ValueError("Please enter a Groq API key.")

    # Load the text using LangChain.
    with TemporaryDirectory() as folder:
        file_path = Path(folder) / "policy.txt"
        file_path.write_text(policy_text, encoding="utf-8")
        docs = TextLoader(str(file_path), encoding="utf-8").load()

    # Keep each policy together so its notice period and approval rule stay intact.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template("""
Answer the question using only the context below.
For questions with multiple parts, answer each part and include the relevant
notice periods and approval requirements. Do not invent any policy.
If the context does not contain the requested information, clearly say that
the provided document does not provide it. Keep the answer concise.

Context: {context}
Question: {question}
Answer:
""")
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0,
    )
    chain = prompt | llm | StrOutputParser()

    questions = [
        'How many paid annual leave days does a full-time employee receive per year?',
        'An employee wants to work remotely on Wednesday and take leave on Friday. What do the company policies say about these two requests?',
        "What is the company's parental leave policy?",
    ]

    outputs = []
    for i, question in enumerate(questions, start=1):
        retrieved_docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        answer = chain.invoke({"context": context, "question": question})

        print(f"\nQuestion {i}: {question}")
        print(f"Retrieved {len(retrieved_docs)} chunks")
        print(answer)
        outputs.append((question, answer))

    # Save the actual answers as comments in this file for submission.
    path = Path(__file__).resolve()
    marker = "# Test outputs"
    code = path.read_text(encoding="utf-8").rsplit("\n" + marker, 1)[0]
    comments = [marker]
    for i, (question, answer) in enumerate(outputs, start=1):
        comments.extend(["#", f"# Q{i}: {question}"])
        comments.extend("# " + line for line in answer.splitlines())
    path.write_text(code + "\n" + "\n".join(comments) + "\n", encoding="utf-8")
    print(f"\nOutputs saved at the end of {path.name}. Download this updated file.")


if __name__ == "__main__":
    main()

# Test outputs
#
# Q1: How many paid annual leave days does a full-time employee receive per year?
# A full-time employee receives **18 paid annual leave days per calendar year**.
#
# Q2: An employee wants to work remotely on Wednesday and take leave on Friday. What do the company policies say about these two requests?
# **Remote work on Wednesday**  
# - Allowed (employees may work remotely on Wednesdays).  
# - Must email the manager **at least one working day in advance** (by Tuesday).  
# - Remote work is **subject to manager approval**.
# 
# **Leave on Friday**  
# - Treated as annual leave (employees have 18 paid days per year).  
# - Must submit the leave request to the manager **at least 3 calendar days before** the Friday (by the preceding Tuesday).  
# - Leave is granted only after **manager confirmation**.
#
# Q3: What is the company's parental leave policy?
# The provided document does not contain any information about a parental leave policy.
