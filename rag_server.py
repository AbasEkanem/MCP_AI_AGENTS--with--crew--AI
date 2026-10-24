from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
)
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
import os
from dotenv import load_dotenv

# MCP server
from mcp.server.fastmcp import FastMCP

# MCP server instance
server_name = "rag_tool_server"
mcp_rag_server_instance = FastMCP(server_name)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Define the path to save/load your FAISS index
FAISS_INDEX_PATH = "faiss_index"  


class Rag_mcp_tool:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model = "gemini-1.5-flash",
            temperature = 0.2,
            api_key = os.getenv("GOOGLE_API_KEY")
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model = "models/embedding-001",
            api_key = os.getenv("GOOGLE_API_KEY")
        )
        # Load existing FAISS index, or build and save new one
        if os.path.exists(FAISS_INDEX_PATH) and os.path.isdir(FAISS_INDEX_PATH):
            print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
            self.raw_vector_store = FAISS.load_local(
                FAISS_INDEX_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print("FAISS index loaded successfully.")
        else:
            print("FAISS index not found. Building and saving new index...")
            urls = [
                "http://acceleratez.com.ng/",
                "http://acceleratez.com.ng/about",
                "http://acceleratez.com.ng/courses/",
                "http://acceleratez.com.ng/blog",
                "http://acceleratez.com.ng/community",
                "http://acceleratez.com.ng/contact",
            ]
            loaded_docs = self.webLoader(urls)
            split_docs = self.split_doc(loaded_docs)
            self.raw_vector_store = FAISS.from_documents(split_docs, self.embeddings)
            self.raw_vector_store.save_local(FAISS_INDEX_PATH)
            print(f"FAISS index built and saved to {FAISS_INDEX_PATH}.")

        # Use retriever from the loaded/created FAISS index
        self.store_data = self.raw_vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # Prepare prompt template
        self.prompt = self.prompt_template()

        # Build the retrieval-augmented chain
        self.chain = self.mcp_rag_chain(self.store_data, self.prompt, self.format_docs)

    def webLoader(self, urls):
        loader = WebBaseLoader(urls)
        raw_data = loader.load()
        # Ensure all data are Document instances
        return [
            data if isinstance(data, Document) else Document(page_content=str(data), metadata={})
            for data in raw_data
        ]

    def split_doc(self, data):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(data)
        return chunks

    def format_docs(self, chunks):
        # Join all page_content text of the retrieved documents
        return "\n\n".join(chunk.page_content for chunk in chunks)

    def prompt_template(self):
        return ChatPromptTemplate.from_messages(
            [SystemMessagePromptTemplate.from_template(
            """You are a high level retrieval agent, that retrieves documents from web pages if the knowledge source is gotten by scrapping webpages or loaded from pdf loaders, adhere strictly to the instructions below:
- retrieve the relevant information from the webpages and answer frequently asked questions based on the retrieved documents.
- look up the vector database and answer questions based on the information provided in the context delimited by triple backticks.
- If you do not know the answer to the questions say you do not know and avoid hallucinations.
context:```{context}```
"""
                ),
                HumanMessagePromptTemplate.from_template("questions:{user_query}"),
            ]
        )

    def mcp_rag_chain(self, retriever, prompt_template, format_docs):
        chain_from_docs = (
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        return (
            RunnableParallel(
                {
                    "context": lambda x: retriever.get_relevant_documents(x["user_query"]),
                    "user_query": RunnablePassthrough(),
                }
            ).assign(answer=chain_from_docs)
        )

    def rag_mcp_query(self, user_query: str) -> str:
        response = self.chain.invoke({"user_query": user_query})
        return response["answer"]


# Instantiate the RAG tool class (loads or builds index on import)
mcp_rag_class_instance = Rag_mcp_tool()


@mcp_rag_server_instance.tool()
def RAG_Pipeline_tool(user_query):
    """Use this tool to look up relevant information and retrieve data based on the user query."""
    return mcp_rag_class_instance.rag_mcp_query(user_query)


if __name__ == "__main__":
    print("RAG MCP Server starting (FastMCP with Gemini LLM)...")
    mcp_rag_server_instance.run(transport="stdio")
    print("RAG MCP Server stopped.")
