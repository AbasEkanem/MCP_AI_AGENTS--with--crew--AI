from crewai import Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter
from mcp.client.stdio import StdioServerParameters
from dotenv import load_dotenv
import os

# Initialize the LLM
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
from crewai import LLM
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key=GOOGLE_API_KEY
)

# Create MCP server parameters (ensure rag_server.py path is correct)
mcp_rag_server_params = StdioServerParameters(
    command="python",
    args=["rag_server.py"],
    env=os.environ.copy(), 
)

with MCPServerAdapter(mcp_rag_server_params) as tools:
    # Debug print all available tools
    # print(f"Available tools: {[tool.name for tool in tools]}")
    # to check if the client is connected to the server.
    print(f"Available tools from MCP server: {[tool.name for tool in tools]}")
    print(f"the mcp client has successfully connected to the MCP server using the MCP server adapter:stdio")

    # Try to find the RagPipeline tool
    try:
        rag_tool = next(tool for tool in tools if tool.name == "RAG_Pipeline_tool")
    except StopIteration:
        raise RuntimeError("RagPipeline tool not found in MCP server tools")

    # Define your agent, task, crew as before
    def retriever_agent():
        return Agent(
            role="an expert in retrieving relevant information from the vector database, base on the user query",
            goal="use the rag_tool to retrieve relevant information from the vector database.",
            backstory=(
                "with your long standing years and expertise in retrieving essential data, "
                "you know that there are valuable information stored in the vector database, "
                "and you know how to expertly retrieve relevant data that best suits the users needs."
            ),
            verbose=True,
            llm=llm
        )
    
    def retriever_agent_task(agent_instance):
        return Task(
            description=f"use the provided tool to look up the vector database and retrieve the relevant information base on the user query",
            expected_output="a detailed information base on the user query, found on the database.",
            tools=[rag_tool],
            agent=agent_instance,
            llm=llm
        )
    
    def retriever_agent_crew(agent_instance, task_instance):
        return Crew(
            agents=[agent_instance],
            tasks=[task_instance],
            process=Process.sequential,
            verbose=True,
            llm=llm
        )
    
    agent_instance = retriever_agent()
    task_instance = retriever_agent_task(agent_instance)
    crew_instance = retriever_agent_crew(agent_instance, task_instance)

    # Interaction loop with error handling
    while True:
        try:
            user_query = input("\n Ask me anything about Acceleratez Hub (or type 'exit' to quit): ")
            if user_query.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            result = crew_instance.kickoff({"user_query": user_query})
            print("\nResponse:\n", result)

        except Exception as e:
            print(f"Error during query processing: {e}")
