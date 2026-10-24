from crewai import Agent, Crew, Task, Process
from crewai_tools import MCPServerAdapter
from mcp.client.stdio import StdioServerParameters
# from mcp.client.sse import sse_client
# from mcp.client.streamable_http import streamablehttp_client
import os
from dotenv import load_dotenv
# Initialize the LLM
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
from crewai import LLM
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key=GOOGLE_API_KEY
)

# create a multiserver params
multiserver_params = StdioServerParameters(
        command = "python",
        args = ["multi_server.py"]
    )

# defining the python context manager using the "with" keyword

with MCPServerAdapter(multiserver_params) as mcp_tools:
    print(f"aggregate tools are:{[tool.name for tool in mcp_tools]}")
    print("the client is connected to the server")
    # defining the agent, tasks and crew
    def multiServer_agent():
        return Agent(
            role="A maths expert specializing in addition, subtraction and multiplication",
            goal="Utilize tools from local Stdio, remote SSE, and remote HTTP MCP servers to perform math calculations",
            backstory=(
                "with your long standing years and expertise as a math assistant."
                "An AI agent capable of leveraging a diverse set of tools from multiple sources to perform mathmatical calculations."
            ),
            tools= [mcp_tools],
            verbose=True,
            llm=llm
        )
    
    def multiserver_agent_task(multiServer_agent):
        return Task(
            description= "use the provided tools in stdio server to perform some mathematical calculations.",
            expected_output= "well calculated result of mathematical calculations performed using the tools in the different servers.",
            tools=[mcp_tools],
            agent=multiServer_agent,
            llm=llm
        )
    
    def multiServer_agent_crew(multiServer_agent, multiserver_agent_task):
        return Crew(
            agents=[multiServer_agent],
            tasks=[multiserver_agent_task],
            process=Process.sequential,
            verbose=True,
            llm=llm
        )
    
    agent = multiServer_agent()
    task = multiserver_agent_task(agent)
    crew = multiServer_agent_crew(agent, task)

    # Interaction loop with error handling
    while True:
        try:
            user_query = input("\nperform your maths calculations here (or type 'exit' to quit): ")
            if user_query.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            result = crew.kickoff({"user_query": user_query})
            print("\nResponse:\n", result)

        except Exception as e:
            print(f"Error during query processing: {e}")

