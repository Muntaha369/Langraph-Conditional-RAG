r"""
================================================================================
 CONDITIONAL GRAPH WORKFLOW IN LANGGRAPH  (Python)
 A "Conditional RAG" example driven by an OpenRouter FREE model.
================================================================================

WHAT IS A "CONDITIONAL GRAPH WORKFLOW"?
---------------------------------------
A LangGraph workflow is a *state machine* built from:

    * NODES            -> plain Python functions (each does ONE job)
    * EDGES            -> fixed "always go from A to B" connections
    * CONDITIONAL EDGES-> a ROUTER: "look at the current state and decide
                          WHICH node to go to next"

Most simple pipelines are linear (A -> B -> C). A *conditional* workflow is
non-linear: after a node runs, a small function inspects the state and picks
one of several possible branches at RUNTIME. This is exactly what we need when
the correct action is not known in advance.

The example below is a CONDITIONAL RAG. "RAG" = Retrieval Augmented Generation:
fetch relevant documents, then let the LLM answer using them. The catch is that
NOT every question needs retrieval:

    - "What is Acme's refund policy?" -> needs the private knowledge base -> RETRIEVE
    - "What is the capital of France?" -> general knowledge, no docs needed   -> DIRECT

So our graph has TWO possible paths and a router node that chooses between them.

GRAPH SHAPE
-----------
                         +-------------------+
        START  --------> |  route_question   |   <-- NODE 1: the decision maker
                         +---------+---------+
                                   |
                  add_conditional_edges(...)   <-- THE CONDITIONAL EDGE (branch here)
                    /                           \
        "retrieve" /                             \ "direct"
                  v                               v
         +----------------+               +---------------------+
         |    retrieve    |               |  generate_direct    |  <-- answers w/o docs
         +-------+--------+               +----------+----------+
                 |                                   |
                 v                                   |
      +----------------------+                       |
      | generate_with_context|  <-- answers using docs|
      +----------+-----------+                       |
                 |                                   |
                 +---------------+-------------------+
                                 v
                                END

Run it (from the project root, so the .env is found):
    uv run conditional-rag
    # or:
    uv run python -m conditional_rag.conditional_rag

================================================================================
"""

# --------------------------------------------------------------------------
# 1) IMPORTS
# --------------------------------------------------------------------------
# `from __future__ import annotations` lets us write modern type hints like
# `list[str]` on every Python version we support. (Purely cosmetic here.)
from __future__ import annotations

# Standard-library helpers.
import time            # used for tiny sleeps between retries of the free API
from typing import List, TypedDict  # TypedDict describes the SHAPE of our state

# `load_dotenv` reads key=value pairs from a .env file into environment vars.
from dotenv import load_dotenv

# LangChain's OpenRouter chat wrapper. It reads OPENROUTER_API_KEY from the env.
from langchain_openrouter import ChatOpenRouter

# Message objects are what a chat model consumes.
from langchain_core.messages import HumanMessage

# The LangGraph building blocks:
#   StateGraph -> the graph/state-machine container
#   START, END -> the special entry and exit markers of every graph
from langgraph.graph import END, START, StateGraph


# --------------------------------------------------------------------------
# 2) LOAD ENVIRONMENT + CREATE THE LLM
# --------------------------------------------------------------------------
# load_dotenv() searches the current directory (and upwards) for a .env file
# and loads OPENROUTER_API_KEY into os.environ. ChatOpenRouter then picks it up
# automatically - no need to pass the key manually anywhere.
load_dotenv()

# We use OpenRouter's FREE router alias "openrouter/free". OpenRouter forwards
# the request to whichever free model is currently available, so you don't have
# to hard-code a specific model id. temperature=0 keeps answers as consistent as
# possible (important for our tiny YES/NO classifier node below).
llm = ChatOpenRouter(model="openrouter/free", temperature=0)


# --------------------------------------------------------------------------
# 3) A TINY "KNOWLEDGE BASE" + MOCK RETRIEVER
# --------------------------------------------------------------------------
# In a real RAG app this would be a vector store (FAISS/Chroma/pgvector). To
# keep the example focused on the CONDITIONAL GRAPH mechanics we use a plain
# list of private documents and a naive keyword-overlap "retriever". The graph
# code is identical either way - you would only swap out the retrieve() body.
KNOWLEDGE_BASE: List[dict] = [
    {
        "id": "refund-policy",
        "text": "Acme's refund policy lets any customer request a full refund "
        "within 30 days of purchase, no questions asked.",
    },
    {
        "id": "pricing",
        "text": "Acme offers two plans: Starter at $19 per month and Premium "
        "at $49 per month. Premium includes priority support.",
    },
    {
        "id": "company-facts",
        "text": "Acme was founded in 2015 by Jane Doe and is headquartered in "
        "Kathmandu, Nepal.",
    },
    {
        "id": "support-hours",
        "text": "Acme's customer support is available Monday to Friday, from "
        "9am to 6pm NPT, by email at support@acme.example.",
    },
]


# --------------------------------------------------------------------------
# 4) THE STATE (the data that flows through the graph)
# --------------------------------------------------------------------------
# Every node receives the current state and returns a PARTIAL update (a dict
# with only the keys it changed). LangGraph merges that partial dict back into
# the running state, so later nodes can read what earlier nodes produced.
#
# A TypedDict gives us type hints + a clear, self-documenting schema. Keys are
# optional-by-omission: a node does not have to return all of them.
class RAGState(TypedDict, total=False):
    question: str          # the user's incoming question
    needs_retrieval: bool  # the router's decision (True = fetch documents)
    route: str             # "retrieve" or "direct" -> used by the conditional edge
    documents: List[str]   # the retrieved context passed to the generator
    answer: str            # the final answer we show the user
    path_taken: str        # human-readable trace of the branch (for learning)


# ==========================================================================
# 5) NODE FUNCTIONS
# --------------------------------------------------------------------------
# Each node is just a Python function: State -> partial State.
# ==========================================================================


def _ask_llm(prompt: str, retries: int = 6) -> str:
    """Call the free model and return its text.

    Free models are occasionally rate-limited or briefly unavailable, so we
    retry a few times with a short back-off. This helper is NOT part of the
    graph logic - it just makes the demo reliable.
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return str(response.content).strip()
        except Exception as exc:  # noqa: BLE001 - demo: surface any API error
            last_error = exc
            time.sleep(2 * (attempt + 1))  # 2s, 4s, 6s...
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_error}")

def route_question(state: RAGState) -> RAGState:
    """NODE 1 - the ROUTER / decision maker.

    This is the node whose OUTPUT determines which branch we take. It asks the
    LLM a single, tightly-scoped yes/no question, then stores the decision in
    the state under two keys:

        needs_retrieval : bool          (the semantic answer)
        route           : "retrieve" / "direct"  (the branch key the edge reads)
    """
    question = state["question"] #type:ignore

    # We constrain the model to answer with exactly one word so parsing is easy.
    classifier_prompt = (
        "You are a router for a customer-support assistant with access to "
        "Acme's PRIVATE internal knowledge base (policies, pricing, company "
        "facts, support hours) and to general world knowledge.\n\n"
        "Decide: does answering the question REQUIRE looking up the private "
        "internal knowledge base?\n"
        "- Answer YES if it asks about Acme, its policies, pricing, or staff.\n"
        "- Answer NO if it is a general question answerable by world knowledge "
        "(e.g. science, history, math, writing, coding).\n\n"
        f"Question: {question}\n"
        "Answer with exactly one word, YES or NO:"
    )

    decision = _ask_llm(classifier_prompt).upper()

    # Simple, forgiving parse of the model's one-word answer.
    needs_retrieval = decision.startswith("Y")

    # A small keyword fallback keeps the demo correct even if the free model
    # returns something unexpected (e.g. an empty string on a bad day).
    if "Y" not in decision and "N" not in decision:
        needs_retrieval = any(
            kw in question.lower()
            for kw in ("acme", "refund", "price", "pricing", "plan", "support",
                       "policy", "company", "founded", "headquarter")
        )

    return {
        "needs_retrieval": needs_retrieval,
        # This string is the VALUE the conditional edge will read next.
        "route": "retrieve" if needs_retrieval else "direct",
    }


def retrieve(state: RAGState) -> RAGState:
    """NODE 2a - fetch relevant documents from the knowledge base.

    Only reached when the router chose the "retrieve" branch. We score each
    document by how many query words it shares, and keep the best ones. A real
    system would embed the query and do a vector similarity search here.
    """
    question_words = set(state["question"].lower().split()) #type:ignore

    scored: list[tuple[int, str]] = []
    for doc in KNOWLEDGE_BASE:
        doc_words = set(doc["text"].lower().split())
        overlap = len(question_words & doc_words)
        if overlap > 0:
            scored.append((overlap, doc["text"]))

    # Highest overlap first; keep at most the top 2 as context.
    scored.sort(key=lambda pair: pair[0], reverse=True)
    documents = [text for _, text in scored[:2]]

    # If nothing matched, fall back to the whole KB so the LLM still has context.
    if not documents:
        documents = [doc["text"] for doc in KNOWLEDGE_BASE]

    return {"documents": documents}


def generate_with_context(state: RAGState) -> RAGState:
    """NODE 2b - answer using the retrieved documents (grounded generation)."""
    context = "\n".join(f"- {doc}" for doc in state["documents"]) #type:ignore

    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['question']}" #type:ignore
    )
    answer = _ask_llm(prompt)

    return {
        "answer": answer,
        "path_taken": "route -> retrieve -> generate_with_context",
    }


def generate_direct(state: RAGState) -> RAGState:
    """NODE 2c - answer from the model's own knowledge (no retrieval)."""
    answer = _ask_llm(state["question"]) #type:ignore

    return {
        "answer": answer,
        "path_taken": "route -> generate_direct (no retrieval needed)",
    }


# ==========================================================================
# 6) BUILD THE GRAPH (this is where the CONDITIONAL WORKFLOW lives)
# ==========================================================================
# `StateGraph(RAGState)` creates an empty state machine whose shared memory is
# our RAGState dict.
graph_builder = StateGraph(RAGState)

# --- Register the nodes -----------------------------------------------------
# A node is just a name -> function mapping. The name is what edges refer to.
graph_builder.add_node("route_question", route_question)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate_with_context", generate_with_context)
graph_builder.add_node("generate_direct", generate_direct)

# --- Fixed edge: the entry point -------------------------------------------
# Every run starts by executing the router node.
graph_builder.add_edge(START, "route_question")

# --- THE CONDITIONAL EDGE (the heart of this example) ----------------------
# add_conditional_edges(source, path_fn, path_map) means:
#
#   * source   = the node we branch FROM ("route_question")
#   * path_fn  = a function that receives the current state and returns a KEY
#                telling LangGraph where to go next. Our router stored that key
#                in state["route"] ("retrieve" or "direct"), so we just read it.
#   * path_map = a dict translating each possible key into a real node name.
#
# In plain English: "After route_question runs, call the lambda to read
# state['route']; if it says 'retrieve' go to the retrieve node, and if it says
# 'direct' go to the generate_direct node."
#
# NOTE: only ONE of these two branches runs per invocation - the graph does not
# run nodes that the router did not select.
graph_builder.add_conditional_edges(
    "route_question",                       # branch FROM this node
    lambda state: state["route"],           # the router: state -> branch key
    {                                       # branch key -> destination node
        "retrieve": "retrieve",
        "direct": "generate_direct",
    },
)

# --- Fixed edges that complete each branch ---------------------------------
# The retrieval branch has one extra step before it can finish...
graph_builder.add_edge("retrieve", "generate_with_context")
graph_builder.add_edge("generate_with_context", END)

# ...while the direct branch finishes immediately.
graph_builder.add_edge("generate_direct", END)

# Compile turns the builder into a runnable graph object called an "app".
app = graph_builder.compile()


# ==========================================================================
# 7) RUN THE WORKFLOW
# ==========================================================================
# We deliberately mix questions that need private docs with general ones so you
# can watch the conditional edge send each query down a different path.
SAMPLE_QUESTIONS = [
    "What is Acme's refund policy?",          # -> RETRIEVE branch
    "How much does the Premium plan cost?",   # -> RETRIEVE branch
    "What is the capital of France?",         # -> DIRECT branch
    "Write a one-sentence motivational quote.",  # -> DIRECT branch
]


def answer(question: str) -> RAGState:
    """Run one question through the compiled graph and return the final state."""
    # .invoke() seeds the graph with an INITIAL state. We only set `question`;
    # the other keys are filled in by the nodes as the graph executes.
    return app.invoke({"question": question}) #type:ignore


def main() -> None:
    print("=" * 72)
    print("LangGraph CONDITIONAL RAG workflow  (OpenRouter free model)")
    print("=" * 72)

    for question in SAMPLE_QUESTIONS:
        print(f"\nQ: {question}")

        result = answer(question)

        # `result` is the FULL final state, so we can explain WHICH path ran and
        # whether documents were fetched - very useful for debugging a graph.
        decision = "RETRIEVE" if result.get("needs_retrieval") else "DIRECT (skip retrieval)"
        print(f"  router decision : {decision}")
        print(f"  path taken      : {result.get('path_taken')}")
        if result.get("documents"):
            print(f"  docs retrieved  : {len(result['documents'])}") #type:ignore
        print(f"  answer          : {result.get('answer')}")

    print("\n" + "=" * 72)
    print("Done. Notice how the SAME graph sent different questions down")
    print("different branches - that is a conditional workflow in LangGraph.")
    print("=" * 72)


if __name__ == "__main__":
    main()

