import os
import pickle
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# --- 1. SET UP THE ENVIRONMENT ---
load_dotenv()
print("✅ Environment variables loaded and LangSmith tracing is configured.")

DATA_DIR = "data"

# --- 2. LOAD PRE-PROCESSED DATA ---
print("Loading pre-processed code elements and summaries...")
try:
    with open(os.path.join(DATA_DIR, "code_elements.pkl"), "rb") as f:
        code_elements = pickle.load(f)
    with open(os.path.join(DATA_DIR, "summaries.pkl"), "rb") as f:
        summaries = pickle.load(f)
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Pre-processed data not found.")
    print("Please run `python build_code_graph.py` first.")
    exit()

# --- 3. CREATE MULTI-VECTOR RETRIEVERS ---
# We are creating two distinct vector stores:
# 1. Summary Vector Store: For high-level, conceptual searches.
# 2. Code Vector Store: For retrieving the actual implementation details.

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

print("Creating vector stores...")
# Create the vector store for summaries
summary_vector_store = FAISS.from_documents(summaries, embeddings)
# Create the vector store for the full code elements
code_vector_store = FAISS.from_documents(code_elements, embeddings)
print("✅ Vector stores created.")

# --- 4. THE MULTI-STEP RETRIEVAL & GENERATION LOGIC ---
# This is the core of our advanced RAG pipeline.

# Initialize the final response generation LLM
llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Define the final prompt template
prompt_template = """
You are an expert AI programming assistant. Your task is to answer a question about a codebase,
using the provided context. The context contains relevant code snippets from multiple files.

Be detailed and thorough in your answer. If the context includes function or class definitions,
explain their purpose, arguments, and return values. If the question involves a process,
explain the sequence of operations, referencing the code.

Here is the context:
---
{context}
---

Here is the user's question:
---
{question}
---

Your answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
llm_chain = LLMChain(llm=llm, prompt=prompt)


def answer_question(question, k=5):
    """
    The main function to answer a question using our multi-retrieval strategy.
    """
    print(f"\n🤔 Processing question: {question}")
    
    # STEP 1: Search the summary vector store.
    # This gives us the most relevant functions/classes at a high level.
    print(f"  1. Searching for top {k} relevant summaries...")
    relevant_summary_docs = summary_vector_store.similarity_search(question, k=k)
    
    # For debugging: print the files of the retrieved summaries
    retrieved_files = {doc.metadata['file_path'] for doc in relevant_summary_docs}
    print(f"     ... Found summaries from files: {retrieved_files}")

    # STEP 2: Retrieve the actual code for the top results.
    # We use the metadata from the summaries to pinpoint which full code elements to fetch.
    # A simple and effective way is to just search the code vector store with the original question.
    # The vector search will find the code chunks that are semantically closest to the question.
    print(f"  2. Retrieving the corresponding full code snippets...")
    relevant_code_docs = code_vector_store.similarity_search(question, k=k)
    
    # STEP 3: Combine and format the context.
    print("  3. Combining context and generating the final answer...")
    context_str = ""
    for doc in relevant_code_docs:
        context_str += f"// FILE: {doc.metadata['file_path']}\n\n{doc.page_content}\n\n---\n\n"
        
    # STEP 4: Generate the final answer.
    response = llm_chain.invoke({"context": context_str, "question": question})
    
    return response['text']


# --- 5. ASK QUESTIONS ---
if __name__ == "__main__":
    # Question that failed in Phase 2 because it crosses file boundaries
    cross_file_question = "How does the Session object in `sessions.py` use the `get_netrc_auth` utility from `utils.py` to prepare a request?"
    answer = answer_question(cross_file_question)
    print("\n" + "="*50)
    print(f"❓ QUESTION: {cross_file_question}")
    print(f"\n🤖 AI ASSISTANT ANSWER:\n{answer}")
    print("="*50)

    # A new, high-level question
    high_level_question = "What are the main ways to provide authentication in the requests library?"
    answer = answer_question(high_level_question, k=8) # Use more results for broad questions
    print("\n" + "="*50)
    print(f"❓ QUESTION: {high_level_question}")
    print(f"\n🤖 AI ASSISTANT ANSWER:\n{answer}")
    print("="*50)
    
    print("\n--- End of Phase 3 ---")