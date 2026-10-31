from langchain_classic.chains.qa_with_sources.map_reduce_prompt import (
    COMBINE_PROMPT as DEFAULT_COMBINE_PROMPT,
)
from langchain_classic.chains.qa_with_sources.map_reduce_prompt import (
    QUESTION_PROMPT as DEFAULT_QUESTION_PROMPT,
)
from langchain_core.prompts import PromptTemplate

# Step 1: Extract relevant information from each chunk
QUESTION_PROMPT = PromptTemplate(
    template=(
        "You are analyzing a real estate article.\n"
        "Extract only information relevant to the user's question.\n"
        "Preserve important figures, dates, and factual statements.\n"
        "Do not invent information.\n\n" + DEFAULT_QUESTION_PROMPT.template
    ),
    input_variables=["context", "question"],
)


# Step 2: Generate the final answer
COMBINE_PROMPT = PromptTemplate(
    template=(
        "You are a professional real estate research assistant.\n"
        "Answer using only the provided article information.\n"
        "Keep the answer clear, concise, and factual.\n"
        "Preserve relevant figures and dates.\n"
        "If the information is insufficient, say so clearly.\n"
        "Include supporting URLs in the SOURCES section.\n\n"
        + DEFAULT_COMBINE_PROMPT.template
    ),
    input_variables=["summaries", "question"],
)


# Step 3: Format retrieved information and its source
EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)
