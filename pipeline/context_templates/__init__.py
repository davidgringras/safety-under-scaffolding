"""
Context templates for long-context ecological validity condition.

Each benchmark has a document template with:
- A document shell (~200 tokens of headers/instructions)
- 12-15 content blocks (~500 tokens each) of domain-relevant filler
- A composition function that places the benchmark prompt in context

Templates are designed to be realistic professional documents that give
Map-Reduce scaffolding something meaningful to chunk, addressing the
reviewer critique that short MC prompts force an inappropriate architecture.
"""

from pipeline.context_templates.truthfulqa import TEMPLATE as TRUTHFULQA_TEMPLATE
from pipeline.context_templates.bbq import TEMPLATE as BBQ_TEMPLATE
from pipeline.context_templates.sycophancy import TEMPLATE as SYCOPHANCY_TEMPLATE
from pipeline.context_templates.xstest import TEMPLATE as XSTEST_TEMPLATE

TEMPLATES: dict[str, dict] = {
    "truthfulqa": TRUTHFULQA_TEMPLATE,
    "bbq": BBQ_TEMPLATE,
    "sycophancy": SYCOPHANCY_TEMPLATE,
    "xstest_orbench": XSTEST_TEMPLATE,
}
