# swarmscope documentation

- [Design brief](plan.md): the plan this implementation follows.
- [Adapters overview](adapters.md): what each adapter can and cannot tell you about lineage.
- Integration guides:
  - [OpenAI SDK and Agents SDK](integrations/openai.md)
  - [LangChain and LangGraph](integrations/langchain.md)
  - [CrewAI](integrations/crewai.md)
- [Storage backends](storage.md): SQLite, DuckDB, Postgres/pgvector; sharing a store across processes.
- [Validation protocol](validation.md) and [results](results.md): reference workloads and the value-vs-k curves.
