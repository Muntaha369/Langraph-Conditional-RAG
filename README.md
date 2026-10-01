# Conditional_RAG — Conditional Graph Workflow in LangGraph

A runnable, heavily-commented example of a **conditional graph workflow** built
with [LangGraph](https://github.com/langchain-ai/langgraph) in Python. It runs on
an **OpenRouter free model** and is managed with [uv](https://docs.astral.sh/uv/).

## The idea

The example is a **Conditional RAG**: instead of *always* retrieving documents,
a **router node** decides at runtime whether retrieval is needed.

```
                              +-------------------+
             START  --------> |  route_question   |   router decides
                              +---------+---------+
                                        |
                        add_conditional_edges(...)      <- the branching point
                          /                          \
              "retrieve" /                            \ "direct"
                        v                              v
               +----------------+            +---------------------+
               |    retrieve    |            |  generate_direct    |
               +-------+--------+            +----------+----------+
                       |                                |
                       v                                |
            +----------------------+                    |
            | generate_with_context|                    |
            +----------+-----------+                    |
                       |                                |
                       +---------------+----------------+
                                       v
                                      END
```

- **"What is Acme's refund policy?"** → `route_question` → **retrieve** → grounded answer
- **"What is the capital of France?"** → `route_question` → **direct** → answer from world knowledge

## Files

| File | Purpose |
| --- | --- |
| `src/conditional_rag/conditional_rag.py` | The whole example (state, nodes, conditional edge, `main()`). |
| `pyproject.toml` | uv-managed dependencies and the `conditional-rag` entry point. |
| `.env` | Holds `OPENROUTER_API_KEY` (git-ignored). |

## Setup

Dependencies are already installed, but to recreate the environment:

```bash
uv sync
```

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=sk-or-v1-...
```

## Run

From the project root (so `.env` is found):

```bash
uv run conditional-rag
# or
uv run python -m conditional_rag.conditional_rag
```

## Key LangGraph concepts shown

1. **State** — a `TypedDict` (`RAGState`) that flows through the graph; each node
   returns a *partial* update that LangGraph merges in.
2. **Nodes** — plain functions (`route_question`, `retrieve`,
   `generate_with_context`, `generate_direct`).
3. **Fixed edges** — `add_edge(START, ...)`, `add_edge("retrieve",
   "generate_with_context")`, `add_edge(..., END)`.
4. **Conditional edge** — `add_conditional_edges("route_question", lambda s:
   s["route"], {"retrieve": "retrieve", "direct": "generate_direct"})`. The
   router function reads the state and picks the next node at runtime.

# Langraph
