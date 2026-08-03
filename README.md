Claude (Anthropic) was used as an AI coding assistant throughout this
project. Specifically:

- The initial architecture, graph wiring, node implementations, knowledge
  base documents, verification heuristics, and automated tests were
  scaffolded by Claude based on the assignment brief.
- I then set up and ran the project myself, locally, with the real
  Hugging Face models (`sentence-transformers/all-MiniLM-L6-v2` and
  `google/flan-t5-small`) on my own machine.
- Running it for real surfaced two genuine bugs that weren't visible in
  offline/mock testing: (1) the small local generation model didn't
  reliably follow the exact `[Source: ...]` citation format, causing
  unnecessary verification failures, and (2) the model occasionally
  repeated phrases/sentences in longer answers, a known beam-search
  degeneration issue with small models. Both were diagnosed from my
  actual terminal output and fixed with Claude's help — (1) by having the
  code attach the citation deterministically instead of relying on the
  model to format it correctly, and (2) by adding a repetition penalty
  and n-gram blocking to the generation call.
- I reviewed, ran, and understand the resulting code, including the
  graph's routing logic, the verification checks, and the reasoning
  behind each fix, and can explain the full workflow end to end.