import os
import pickle
import json
from dotenv import load_dotenv
from typing import List, Dict, TypedDict, Optional

# --- LangChain & LangGraph Imports ---
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.pydantic_v1 import BaseModel, Field
from langchain.vectorstores import FAISS
from langgraph.graph import StateGraph, END

# --- 1. SETUP & LOAD DATA ---
load_dotenv()
print("✅ Environment variables loaded.")

DATA_DIR = "data"
print("Loading pre-processed data...")
try:
    with open(os.path.join(DATA_DIR, "code_elements.pkl"), "rb") as f:
        code_elements = pickle.load(f)
    with open(os.path.join(DATA_DIR, "summaries.pkl"), "rb") as f:
        summaries = pickle.load(f)
    print("✅ Data loaded.")
except FileNotFoundError:
    print("❌ Pre-processed data not found. Please run `build_code_graph.py` first.")
    exit()

# --- 2. INITIALIZE MODELS AND RETRIEVERS (GEMINI-ONLY) ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
summary_vector_store = FAISS.from_documents(summaries, embeddings)
summary_retriever = summary_vector_store.as_retriever(k=5)

code_vector_store = FAISS.from_documents(code_elements, embeddings)
code_retriever = code_vector_store.as_retriever(k=5)

# LLM for generating the free-form answer.
# We use the standard `GoogleGenerativeAI` for this.
generator_llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# LLM for the structured evaluation task.
# We use `ChatGoogleGenerativeAI` because its API is better suited for tool-calling,
# which is what LangChain uses under the hood for `with_structured_output`.
evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

print("✅ All-Gemini models and retrievers initialized.")


# --- 3. DEFINE THE AGENT'S STATE ---
class AgentState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    evaluation: Optional[Dict]
    iteration: int

# --- 4. DEFINE THE GRAPH NODES ---

def retrieve_node(state: AgentState) -> AgentState:
    """Retrieves relevant code snippets."""
    print("--- 🔄 NODE: RETRIEVE ---")
    question = state["question"]
    relevant_code = code_retriever.get_relevant_documents(question)
    
    for doc in relevant_code:
        doc.page_content = f"// FILE: {doc.metadata['file_path']}\n\n{doc.page_content}"
        
    return {"context": relevant_code, "iteration": state.get("iteration", 0) + 1}

def generate_node(state: AgentState) -> AgentState:
    """Generates an answer using the retrieved context."""
    print("--- 🔄 NODE: GENERATE ---")
    context_str = "\n\n---\n\n".join([doc.page_content for doc in state["context"]])
    
    prompt = PromptTemplate(
        template="""You are an expert AI programming assistant. Answer the user's question based on the provided code context. Be detailed and clear.

        CONTEXT:
        {context_str}

        QUESTION:
        {question}

        ANSWER:""",
        input_variables=["context_str", "question"]
    )
    
    chain = prompt | generator_llm
    answer = chain.invoke({"context_str": context_str, "question": state["question"]})
    return {"answer": answer}

# Pydantic model for the structured output of the evaluator
class Evaluation(BaseModel):
    is_supported: bool = Field(description="True if the answer is fully supported by the context, False otherwise.")
    reasoning: str = Field(description="A brief explanation of why the answer is or is not supported.")

def evaluate_node(state: AgentState) -> AgentState:
    """Evaluates the generated answer against the context for faithfulness."""
    print("--- 🔄 NODE: EVALUATE ---")
    context_str = "\n\n---\n\n".join([doc.page_content for doc in state["context"]])
    
    prompt_str = f"""You are a strict evaluator. Your task is to determine if an AI-generated answer is fully supported by the provided source code context.
    Respond with a JSON object containing two keys: 'is_supported' (a boolean) and 'reasoning' (a string).

    CONTEXT:
    {context_str}

    QUESTION:
    {state['question']}

    AI-GENERATED ANSWER:
    {state['answer']}
    
    EVALUATION (JSON):"""
    
    # We use `.with_structured_output` which leverages Gemini's tool-calling
    # to force the model to return a JSON object matching our Pydantic schema.
    evaluator_chain = evaluator_llm.with_structured_output(Evaluation)
    
    evaluation_result = evaluator_chain.invoke(prompt_str)
    
    print(f"  > Evaluation result: {evaluation_result.is_supported}. Reason: {evaluation_result.reasoning}")
    return {"evaluation": evaluation_result.dict()}


# --- 5. DEFINE THE GRAPH EDGES ---

def should_continue(state: AgentState) -> str:
    """Conditional edge: decides whether to finish or re-try."""
    print("--- ❔ CONDITIONAL EDGE: SHOULD CONTINUE? ---")
    if state["iteration"] > 2:
        print("  > Max iterations reached. Finishing.")
        return "end"

    if state["evaluation"]["is_supported"]:
        print("  > Answer is supported. Finishing.")
        return "end"
    else:
        print("  > Answer not supported. Re-running retrieval for more context.")
        return "continue"

# --- 6. ASSEMBLE THE GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("evaluate", evaluate_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "evaluate")
workflow.add_conditional_edges(
    "evaluate",
    should_continue,
    {"end": END, "continue": "retrieve"}
)

app = workflow.compile()
print("✅ Pure-Gemini agent graph compiled.")

# --- 7. RUN THE AGENT ---
if __name__ == "__main__":
    question = "Explain how the `requests` library handles proxy settings. Does it check environment variables?"
    
    inputs = {"question": question}
    print(f"\n--- 🚀 RUNNING AGENT on question: '{question}' ---")
    # Stream the events to see the agent's "thoughts" in real-time
    for output in app.stream(inputs, {"recursion_limit": 5}):
        for key, value in output.items():
            # This is a simple way to print the state after each node completes
            print(f"Finished node '{key}'. Current state keys: {list(value.keys())}")
    
    # Get the final response from the last streamed event
    final_state = list(app.stream(inputs, {"recursion_limit": 5}))[-1]
    final_answer = final_state[list(final_state.keys())[0]]['answer']

    print("\n" + "="*50)
    print(f"❓ QUESTION: {question}")
    print(f"\n🤖 FINAL AGENT ANSWER:\n{final_answer}")
    print("="*50)
    print("\n--- End of Phase 4 (Gemini-Only Edition) ---")