import os
import logging

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
    snippet_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# Nodes
def start_research(state: OverallState, config: RunnableConfig) -> dict:
    """
    Read initial configuration and populate the state.
    This is the new entry point for the graph.
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Plumb all relevant config values into the state
    state_update = {
        "initial_search_query_count": configurable.number_of_initial_queries,
        "max_research_loops": configurable.max_research_loops,
        "reasoning_model": configurable.answer_model, # Using answer_model as default
        "output_format": config.get("configurable", {}).get("output_format", "detailed"),
    }
    
    # Override with any specific values passed in the invoke/submit call
    # This ensures values from the frontend take precedence
    if state.get("initial_search_query_count"):
        state_update["initial_search_query_count"] = state["initial_search_query_count"]
    if state.get("max_research_loops"):
        state_update["max_research_loops"] = state["max_research_loops"]
    if state.get("reasoning_model"):
        state_update["reasoning_model"] = state["reasoning_model"]

    # Debug print to confirm
    print(f"--- Initializing state with: {state_update} ---")
    
    return state_update


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question."""
    
    # Configurable is now only used for the model name
    configurable = Configuration.from_runnable_config(config)

    # The initial query count is now already in the state
    initial_query_count = state.get("initial_search_query_count", 3)

    # init Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=initial_query_count,
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    logger.info("[continue_to_web_research] Entry")
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    logger.info("[web_research] Entry")
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Uses the google genai client as the langchain client doesn't return grounding metadata
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    # resolve the urls to short urls for saving tokens and time
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    )
    # Gets the citations and adds them to the generated text
    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    logger.info("[web_research] Output: sources_gathered=%s, web_research_result=%s", sources_gathered, modified_text)
    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    logger.info("[reflection] Entry")
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)
    logger.info("[reflection] Output: is_sufficient=%s, knowledge_gap=%s, follow_up_queries=%s", result.is_sufficient, result.knowledge_gap, result.follow_up_queries)
    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    logger.info("[evaluate_research] Entry")
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        logger.info("[evaluate_research] Decision: finalize_answer")
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]

# def should_generate_snippet(state: OverallState) -> str:
#     """Determines whether to generate a snippet or end."""
#     # Add this print statement for easy debugging
#     print(f"--- Deciding on output format. Found: {state.get('output_format')} ---")
#     if state.get("initial_search_query_count") < 3:
#         return "generate_snippet"
#     else:
#         logger.info("[should_generate_snippet] Decision: END")
#         return "__end__" # Return the string "__end__"

def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    logger.info("[finalize_answer] Entry")
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model, default to Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)

    # Create a variable for the processed content
    processed_content = result.content

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in processed_content:
            processed_content = processed_content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    logger.info("[finalize_answer] Output: processed_content=%s, unique_sources=%s", processed_content, unique_sources)
    output = {
        "messages": [
            AIMessage(
                content=processed_content,
            )
        ],
        "sources_gathered": unique_sources,
    }

    return output


def generate_snippet(state: OverallState, config: RunnableConfig):
    logger.info("[generate_snippet] Entry")
    configurable = Configuration.from_runnable_config(config)

    # The detailed answer is in the last message
    final_answer = state["messages"][-1].content

    # Format the prompt
    formatted_prompt = snippet_instructions.format(final_answer=final_answer)

    # Use a fast model for summarization
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    snippet = llm.invoke(formatted_prompt).content
    logger.info("[generate_snippet] Output: snippet=%s", snippet)

    # Create a new message with the snippet content, replacing the detailed one
    # Note: We are not preserving sources here as snippets don't have them.
    final_snippet_message = AIMessage(content=snippet)

    return {"messages": [final_snippet_message]}


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("start_research", start_research)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)
builder.add_node("generate_snippet", generate_snippet)

# --- UPDATE THE ENTRYPOINT ---
# Set the entrypoint to our new start_research node
builder.add_edge(START, "start_research")
# Now, start_research flows into generate_query
builder.add_edge("start_research", "generate_query")

# The rest of the graph wiring remains the same
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
builder.add_edge("web_research", "reflection")
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# always generate a snippet after finalizing the answer
builder.add_edge("finalize_answer", "generate_snippet")
builder.add_edge("generate_snippet", END)

graph = builder.compile(name="pro-search-agent")