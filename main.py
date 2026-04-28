import os
import json
import re
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_neo4j import Neo4jGraph, Neo4jVector, GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer

import streamlit as st


def discover_schema(llm, docs, sample_size=10):
    """Use LLM to infer node types and relationships from document sample."""
    # Sample from beginning, middle, and end for better coverage
    if len(docs) <= sample_size:
        sample_docs = docs
    else:
        indices = [int(i * len(docs) / sample_size) for i in range(sample_size)]
        sample_docs = [docs[i] for i in indices]

    sample_text = "\n\n".join([doc.page_content for doc in sample_docs])

    prompt = f"""Analyze this text and identify the key entity types and relationships present.

Text:
{sample_text}

Respond ONLY with valid JSON in this exact format:
{{
  "nodes": ["EntityType1", "EntityType2", "EntityType3"],
  "relationships": ["RELATIONSHIP_ONE", "RELATIONSHIP_TWO"]
}}

Rules:
- Node types should be singular PascalCase nouns (e.g. "Person", "Company", "Product")
- Relationships should be SCREAMING_SNAKE_CASE verbs (e.g. "WORKS_FOR", "LOCATED_IN")
- Only include types clearly present in the text
- Aim for 4-8 node types and 4-10 relationships
- Return ONLY the JSON, no explanation"""

    response = llm.invoke(prompt)
    json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
    if json_match:
        schema = json.loads(json_match.group())
        return schema["nodes"], schema["relationships"]

    raise ValueError("Could not parse schema from LLM response")


def main():
    # ── Page config ──────────────────────────────────────────────────────────
    st.set_page_config(
        layout="wide",
        page_title="Graphy v1",
        page_icon=":graph:"
    )

    st.sidebar.image('4.jpeg', width=200)
    with st.sidebar.expander("Expand Me"):
        st.markdown("""
        This application allows you to upload a PDF file, extract its content into a
        Neo4j graph database, and perform queries using natural language.
        It leverages LangChain and Groq's LLaMA models to generate Cypher queries
        that interact with the Neo4j database in real-time.
        """)
    st.title("Graphy: Realtime GraphRAG App")

    # ── LLM + Embeddings init ─────────────────────────────────────────────────
    if 'llm' not in st.session_state or 'embeddings' not in st.session_state:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY")
        if not gemini_api_key:
            st.sidebar.error("GEMINI_API_KEY not found in .streamlit/secrets.toml")
            return
      

        with st.spinner("Loading embedding model (first run may take a minute)..."):
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=gemini_api_key
            )
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=gemini_api_key
            )   
            st.session_state['llm'] = llm
            st.session_state['embeddings'] = embeddings
            st.sidebar.success("LLM and embeddings ready.")
    else:
        st.sidebar.success("LLM already in session state.")

    llm = st.session_state.get('llm')
    embeddings = st.session_state.get('embeddings')

    if not llm or not embeddings:
        st.error("LLM not initialized. Check your GROQ_API_KEY in secrets.")
        return

    # ── Neo4j connection ──────────────────────────────────────────────────────
    if 'neo4j_connected' not in st.session_state:
        neo4j_url      = st.secrets.get('NEO4J_URL')
        neo4j_username = st.secrets.get('NEO4J_USERNAME')
        neo4j_password = st.secrets.get('NEO4J_PASSWORD')
        neo4j_database = st.secrets.get('NEO4J_DATABASE', 'neo4j')

        if not all([neo4j_url, neo4j_username, neo4j_password]):
            st.sidebar.error(
                "Neo4j secrets not found. Please set NEO4J_URL, NEO4J_USERNAME, "
                "and NEO4J_PASSWORD in .streamlit/secrets.toml"
            )
            return

        try:
            graph = Neo4jGraph(
                url=neo4j_url,
                username=neo4j_username,
                password=neo4j_password,
                database=neo4j_database
            )
            st.session_state['graph']          = graph
            st.session_state['neo4j_url']      = neo4j_url
            st.session_state['neo4j_username'] = neo4j_username
            st.session_state['neo4j_password'] = neo4j_password
            st.session_state['neo4j_database'] = neo4j_database
            st.session_state['neo4j_connected'] = True
            st.sidebar.success("Connected to Neo4j database.")
        except Exception as e:
            st.sidebar.error(f"Failed to connect to Neo4j: {e}")
            return
    else:
        graph          = st.session_state.get('graph')
        neo4j_url      = st.session_state.get('neo4j_url')
        neo4j_username = st.session_state.get('neo4j_username')
        neo4j_password = st.session_state.get('neo4j_password')
        neo4j_database = st.session_state.get('neo4j_database', 'neo4j')

    # ── PDF upload + processing ───────────────────────────────────────────────
    if graph is not None:
        uploaded_file = st.file_uploader("Please select a PDF file.", type="pdf")

        if uploaded_file is not None and 'qa' not in st.session_state:

            # 1. Load & chunk the PDF
            with st.spinner("Loading and chunking PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                loader = PyPDFLoader(tmp_file_path)
                pages  = loader.load_and_split()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size= 1000, chunk_overlap=100 
                )
                docs = text_splitter.split_documents(pages)

                lc_docs = [
                    Document(
                        page_content=doc.page_content.replace("\n", ""),
                        metadata={'source': uploaded_file.name}
                    )
                    for doc in docs
                ]

            # 2. Discover schema dynamically
            with st.spinner("Discovering schema from document..."):
                allowed_nodes, allowed_relationships = discover_schema(llm, lc_docs)

            with st.expander("📊 Discovered Schema"):
                st.write("**Node Types:**", allowed_nodes)
                st.write("**Relationships:**", allowed_relationships)

            # 3. Clear old graph data
            with st.spinner("Clearing existing graph data..."):
                graph.query("MATCH (n) DETACH DELETE n")

            # 4. Build graph from documents
            with st.spinner("Extracting entities and relationships (this may take a while)..."):
                transformer = LLMGraphTransformer(
                    llm=llm,
                    allowed_nodes=allowed_nodes,
                    allowed_relationships=allowed_relationships,
                    node_properties=False,
                    relationship_properties=False
                )
                graph_documents = transformer.convert_to_graph_documents(lc_docs)
                graph.add_graph_documents(graph_documents, include_source=True)

            # 5. Build vector index
            with st.spinner("Building vector index..."):
                try:
                    Neo4jVector.from_existing_graph(
                        embedding=embeddings,
                        url=neo4j_url,
                        username=neo4j_username,
                        password=neo4j_password,
                        database=neo4j_database,
                        node_label=allowed_nodes[0],
                        text_node_properties=["id", "text"],
                        embedding_node_property="embedding",
                        index_name="vector_index",
                        keyword_index_name="entity_index",
                        search_type="hybrid"
                    )
                except Exception as e:
                    st.warning(f"Vector index warning (non-fatal): {e}")

            st.success(f"✅ {uploaded_file.name} processed successfully!")

            # 6. Set up QA chain
            if 'qa' not in st.session_state:
                node_count = graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
                with st.spinner(f"Processing {node_count} nodes..."):
                    if node_count == 0:
                        st.info("Graph is empty. Please upload a PDF to get started.")
                    else:
                        schema = graph.schema
                        template = """
Task: Generate a Cypher statement to query the graph database.
Instructions:
Use only relationship types and properties provided in schema.
Do not use other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include explanations or apologies in your answers.
Do not answer questions that ask anything other than creating Cypher statements.
Do not include any text other than the generated Cypher statement.
Question: {question}"""

                        question_prompt = PromptTemplate(
                            template=template,
                            input_variables=["schema", "question"]
                        )
                        qa = GraphCypherQAChain.from_llm(
                            llm=llm,
                            graph=graph,
                            cypher_prompt=question_prompt,
                            verbose=True,
                            allow_dangerous_requests=True
                        )
                        st.session_state['qa'] = qa
                        st.sidebar.success("QA chain ready.")

        # ── Q&A interface ─────────────────────────────────────────────────────────
        question = st.text_input("Ask a question about your PDF:")
        if question:
            with st.spinner("Generating answer..."):
                try:
                    res = st.session_state['qa'].invoke({"query": question})
                    st.write("\n**Answer:**\n" + res['result'])
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

    else:  # ← this else belongs to "if graph is not None"
        st.warning("Please connect to the Neo4j database before uploading a PDF.")


if __name__ == "__main__":
    main()