from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


query_writer_instructions = """Your goal is to generate sophisticated and diverse web search queries. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.

Format: 
- Format your response as a JSON object with ALL two of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": A list of search queries

Example:

Topic: What revenue grew more last year apple stock or the number of people buying an iphone
```json
{{
    "rationale": "To answer this comparative growth question accurately, we need specific data points on Apple's stock performance and iPhone sales metrics. These queries target the precise financial information needed: company revenue trends, product-specific unit sales figures, and stock price movement over the same fiscal period for direct comparison.",
    "query": ["Apple total revenue growth fiscal year 2024", "iPhone unit sales growth fiscal year 2024", "Apple stock price growth fiscal year 2024"],
}}
```

Context: {research_topic}"""


web_searcher_instructions = """Conduct targeted Google Searches to gather the most recent, credible information on "{research_topic}" and synthesize it into a verifiable text artifact.

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- Conduct multiple, diverse searches to gather comprehensive information.
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings. 
- Only include the information found in the search results, don't make up any information.

Research Topic:
{research_topic}
"""

reflection_instructions = """You are an expert research assistant analyzing summaries about "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration and generate a follow-up query. (1 or multiple).
- If provided summaries are sufficient to answer the user's question, don't generate a follow-up query.
- If there is a knowledge gap, generate a follow-up query that would help expand your understanding.
- Focus on technical details, implementation specifics, or emerging trends that weren't fully covered.

Requirements:
- Ensure the follow-up query is self-contained and includes necessary context for web search.

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing or needs clarification
   - "follow_up_queries": Write a specific question to address this gap

Example:
```json
{{
    "is_sufficient": true, // or false
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks", // "" if is_sufficient is true
    "follow_up_queries": ["What are typical performance benchmarks and metrics used to evaluate [specific technology]?"] // [] if is_sufficient is true
}}
```

Reflect carefully on the Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:

Summaries:
{summaries}
"""

answer_instructions = """Generate a high-quality answer to the user's question based on the provided summaries.

Instructions:
- The current date is {current_date}.
- You are the final step of a multi-step research process, don't mention that you are the final step. 
- You have access to all the information gathered from the previous steps.
- You have access to the user's question.
- Generate a high-quality answer to the user's question based on the provided summaries and the user's question.
- Include the sources you used from the Summaries in the answer correctly, use markdown format (e.g. [apnews](https://vertexaisearch.cloud.google.com/id/1-0)). THIS IS A MUST.

User Context:
- {research_topic}

Summaries:
{summaries}"""
snippet_instructions = """Generate a brief, news-style snippet summarizing the research output below. The snippet should be concise, highlight key findings, market movements, and important news, and be written in the style of the following example:

Example:
Snippet 1- Nifty Pre Open- down 152 pts. Key news- ICICI Bank – Posted Q3 mixed, business momentum weakens; return ratios remain steady. AU Small Finance Bank - FY25 loan growth guidance cut to 20 pct, FY26 growth guidance withheld. Macrotech - Q3 pre-sales jumped 32 pct YoY, debt reduced by Rs. 6.10 bn. Yes Bank - Retail book declined by 3.2 pct, NIM flat YoY and QoQ at 2.4 pct. DLF - Posts best-ever quarterly new sales booking collections up 23 pct YoY. CreditAccess - FY25 guidance revised lower for the second time in 3 months, NIM at 6-quarter low. AB Real Estate - Arm signs JV agreement with Mitsubishi Estate for a project, investment at Rs. 5.6 bn. IDFC First - Q3 below poll, slippage ratio is the highest in 11 quarters. EMS - EBITDA jumped 52 pct, margin at 29 pct vs 23.5 pct YoY Trident - EBITDA declined 18.8 pct, margin at 12.8 pct vs 14.3 pct YoY. Indigo - Ex-forex, earnings in-line, Q4 ASK guidance of 20 pct YoY. NTPC Green - EBITDA declined 2.3 pct, margin at 83.5 pct vs 88.9 pct YoY. JK Cement - Q3 sales volumes up 5 pct, to acquire 60 pct stake in Saifco Cement for Rs. 1.74 bn. NTPC - Muted Q3, capacity addition slow. Results Today – Tata Steel, Bajaj Housing Fin

Instructions:
- Write in a similar style as the example above.
- Focus on the most important findings, news, and market movements from the research output.
- Be concise and use a news-brief format.
- Do not include citations or URLs.
- Output only the snippet text.

Research Output:
{final_answer}
"""

