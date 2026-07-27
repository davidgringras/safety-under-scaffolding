"""
Fixed production framework runners for E9 evaluation.

These implementations address Criticism 5 from the C1 adversarial review:
the original E9 runners tested prompt templates, not actual framework
capabilities. Each runner here uses genuine framework features:

- crewai_runner: Multi-agent crew with Safety Analyst + Reviewer agents,
  delegation enabled, actual CrewAI orchestration.
- langchain_runner: Real LangChain LCEL chain with ChatPromptTemplate,
  ChatLiteLLM model wrapper, and StrOutputParser.
- openai_sdk_runner: OpenAI Agents SDK used for ALL models via the
  external model provider interface.
"""

__version__ = "2.0.0"
