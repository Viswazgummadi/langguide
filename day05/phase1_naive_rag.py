import os
from dotenv import load_dotenv

# LangChain components
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- 1. SET UP THE ENVIRONMENT ---
# Load environment variables from .env file (API keys, etc.)
load_dotenv()
print("✅ Environment variables loaded.")

# Check if the LangSmith keys are set to enable tracing
# This is invaluable for debugging and seeing what the agent is doing.
if os.getenv("LANGCHAIN_TRACING_V2"):
    print("✅ LangSmith tracing is enabled.")
else:
    print("⚠️ LangSmith tracing is not enabled. Set LANGCHAIN_TRACING_V2 to 'true' in .env to enable.")

# --- 2. LOAD THE DATA ---
# We are targeting a single, complex file from our cloned repo.
# 'requests/sessions.py' is a great example as it contains a lot of logic.
file_path = "source_code/requests/src/requests/sessions.py"
loader = TextLoader(file_path, encoding='utf-8')
documents = loader.load()
print(f"✅ Loaded {len(documents)} document from {file_path}.")

# --- 3. SPLIT THE DOCUMENT ---
# We split the loaded document into smaller chunks.
# This is a "naive" approach, just splitting by character count.
# We'll see its limitations soon.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f"✅ Split the document into {len(texts)} chunks.")

# --- 4. EMBED AND STORE ---
# We use Google's embedding model to turn text chunks into vectors.
# These vectors are then stored in FAISS, a local vector store.
# This allows for fast similarity searches.
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(texts, embeddings)
print("✅ Created FAISS vector store from document chunks.")

# --- 5. RETRIEVE AND GENERATE ---
# We create a RetrievalQA chain. This is a standard RAG pattern.
# It retrieves relevant documents from the vector store and then uses an LLM
# to generate an answer based on the retrieved context.
llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # "Stuff" means it crams all retrieved docs into the prompt
    retriever=vector_store.as_retriever()
)
print("✅ Created RetrievalQA chain.")

# --- 6. ASK QUESTIONS ---
# Let's test it with two questions.

# Question 1: Simple, likely to succeed.
print("\n--- Asking a simple question ---")
simple_question = "What is the purpose of the Session class?"
print(f"❓ Question: {simple_question}")
simple_result = qa_chain.invoke({"query": simple_question})
print(f"🤖 Answer: {simple_result['result']}")

print("\n" + "="*50 + "\n")

# Question 2: Complex, likely to fail.
# This question requires understanding multiple functions (cookie handling, request sending, etc.)
print("--- Asking a complex, multi-step question ---")
complex_question = "How are cookies handled across multiple requests made with a Session object?"
print(f"❓ Question: {complex_question}")
complex_result = qa_chain.invoke({"query": complex_question})
print(f"🤖 Answer: {complex_result['result']}")
print("\n--- End of Phase 1 ---")