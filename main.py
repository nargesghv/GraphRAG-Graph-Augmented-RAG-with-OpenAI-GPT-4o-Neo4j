# --- Imports ---
import os
from getpass import getpass

from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI

# --- OpenAI API Key ---
os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key:")

# --- Connect to Neo4j ---
graph = Neo4jGraph(
    url="neo4j://localhost",
    username="neo4j",
    password="password4j"
)

# --- Input Text to Transform ---
graph_text = """
John's title is Director of the Digital Marketing Group.
John works with Jane, whose title is Chief Marketing Officer.
Jane works in the Executive Group. Jane works with Sharon whose title is the Director of Client Outreach.
Sharon works in the Sales Group.
"""

# --- OpenAI GPT-4o-mini (used for both graph generation & Cypher QA) ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- LLM to Graph Transformer ---
llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Title", "Group"],
    allowed_relationships=["TITLE", "COLLABORATES", "GROUP"]
)

documents = [Document(page_content=graph_text)]
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# --- Load Graph to Neo4j ---
graph.add_graph_documents(graph_documents)

# --- Schema for prompting ---
graph.refresh_schema()
schema = graph.get_schema
print(schema)

# --- Cypher Prompt (Few-Shot Examples) ---
examples = [
    {
        "question": "What group is Charles in?",
        "query": "MATCH (p:Person {id: 'Charles'})-[:GROUP]->(g:Group) RETURN g.id"
    },
    {
        "question": "Who does Paul work with?",
        "query": "MATCH (a:Person {id: 'Paul'})-[:COLLABORATES]->(p:Person) RETURN p.id"
    },
    {
        "question": "What title does Rico have?",
        "query": "MATCH (p:Person {id: 'Rico'})-[:TITLE]->(t:Title) RETURN t.id"
    }
]

example_prompt = PromptTemplate.from_template("{query}")

prefix = """Task: Generate a Cypher statement to query a graph database strictly based on the schema and instructions provided.

Instructions:
- Respond with ONE and ONLY ONE query.
- Use provided node and relationship labels and property names from the schema.
- Generate a valid executable Cypher query for the Neo4j database.
- Do not include explanations, context, or any text outside the query.
- Do not include output labels or comments.
- Do not include multiple questions or queries.

Here is the schema information:

{schema}

With all the above information and instructions, generate Cypher query for the following question:

The question is:
"""

cypher_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix="{question}\nCypher query:",
    input_variables=["question", "schema"]
)

# --- QA Prompt Template ---
qa_examples = [
    {
        "question": "What group is Charles in?",
        "context": "[{{'g.id': 'Executive Group'}}]",
        "response": "Charles is in the Executive Group"
    },
    {
        "question": "Who does Paul work with?",
        "context": "[{{'p.id': 'Greg'}}, {{'p2.id': 'Norma'}}]",
        "response": "Paul works with Greg and Norma"
    },
    {
        "question": "What title does Rico have?",
        "context": "[{{'t.id': 'Vice President of Sales'}}]",
        "response": "Vice President of Sales"
    }
]

qa_template = """
Use the provided question and context to create an answer.

Question: {question}

Context: {context}

Use only names, departments, or titles contained within {question} and {context}.
"""

qa_example_prompt = PromptTemplate.from_template("")
qa_prompt = FewShotPromptTemplate(
    examples=qa_examples,
    prefix=qa_template,
    input_variables=["question", "context"],
    example_prompt=qa_example_prompt,
    suffix=""
)

# --- GraphCypherQAChain with OpenAI ---
chain = GraphCypherQAChain.from_llm(
    prompt=cypher_prompt,
    qa_prompt=qa_prompt,
    graph=graph,
    llm=llm,
    exclude_types=["Genre"],
    verbose=True,
    allow_dangerous_requests=True,
    validate_cypher=True
)

# --- Query the Knowledge Graph ---
chain.invoke({"query": "What is John's title?"})
chain.invoke("Who does John collaborate with?")
chain.invoke("What group is Jane in?")
chain.invoke("Who does Jane collaborate with?")
