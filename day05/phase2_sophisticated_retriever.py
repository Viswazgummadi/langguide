import os
from dotenv import load_dotenv

# LangChain components
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain_text_splitters import PythonCodeTextSplitter # <-- The new, smarter splitter
from langchain.vectorstores import FAISS
from langchain.storage import InMemoryStore # <-- To store parent documents
from langchain.retrievers import ParentDocumentRetriever # <-- The new retriever
from langchain.chains import RetrievalQA

# --- 1. SET UP THE ENVIRONMENT ---
load_dotenv()
print("✅ Environment variables loaded and LangSmith tracing is configured.")

# --- 2. LOAD THE DATA ---
# Same file as before for a direct comparison
file_path = "source_code/requests/src/requests/sessions.py"
loader = TextLoader(file_path, encoding='utf-8')
parent_documents = loader.load()
print(f"✅ Loaded {len(parent_documents)} parent document from {file_path}.")

# --- 3. SPLIT THE DOCUMENT... INTELLIGENTLY ---
# This splitter understands Python syntax. It will try to split along
# function and class definitions, keeping them intact.
# These will be our "child" documents.
child_splitter = PythonCodeTextSplitter(chunk_size=1000, chunk_overlap=100)
child_documents = child_splitter.split_documents(parent_documents)
print(f"✅ Split the document into {len(child_documents)} syntactically-aware child chunks.")

# --- 4. EMBED AND STORE WITH PARENT-CHILD STRATEGY ---
# The core idea:
# 1. We search for similarity over the small, specific child chunks.
# 2. We then retrieve the larger, more context-rich parent chunk associated with the best child.

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# The vector store holds the small child documents
vector_store = FAISS.from_documents(child_documents, embeddings)
print("✅ Created FAISS vector store from child chunks.")

# The in-memory store holds the raw parent documents
docstore = InMemoryStore()

# The ParentDocumentRetriever orchestrates the process
retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=docstore,
    child_splitter=child_splitter,
    # We need to add the parent documents to the store
    # so they can be retrieved later.
    parent_splitter=None # The original docs are our parents
)
# This step is crucial. It populates the docstore.
retriever.add_documents(parent_documents, ids=None)
print("✅ Created ParentDocumentRetriever and populated the docstore.")

# --- 5. RETRIEVE AND GENERATE ---
# We use the same QA chain, but now it's powered by our more sophisticated retriever.
llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)
print("✅ Created RetrievalQA chain with the new retriever.")

# --- 6. ASK THE COMPLEX QUESTION AGAIN ---
print("\n" + "="*50 + "\n")
print("--- Asking the same complex question as in Phase 1 ---")
complex_question = "How are cookies handled across multiple requests made with a Session object?"
print(f"❓ Question: {complex_question}")

# We can also ask the chain to return the source documents it used
qa_chain.return_source_documents = True
complex_result = qa_chain.invoke({"query": complex_question})

print(f"🤖 Answer: {complex_result['result']}")

# Let's inspect the source documents to see the improvement
print("\n--- Source Documents Retrieved ---")
for doc in complex_result['source_documents']:
    # The retriever returns the WHOLE FILE because that's our parent document.
    # This gives the LLM massive context.
    print(f"📄 Document from '{doc.metadata['source']}' (showing first 200 chars):")
    print(doc.page_content[:200] + "...")
    
print("\n--- End of Phase 2 ---")