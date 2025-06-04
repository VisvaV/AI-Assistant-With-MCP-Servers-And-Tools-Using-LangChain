import asyncio
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp_use import MCPAgent, MCPClient

async def run_memory_chat():
    """Run a chat using MCPAgent's built-in conversation memory."""

    # Load environment variables for API keys
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

    # Config file path – change this to your config file if needed
    config_file = "browser_config.json"

    print("Initializing chat...")

    # Create MCP client
    print("Creating MCPClient...")
    client = MCPClient.from_config_file(config_file)
    print("MCPClient created.")

    # Create Gemini chat model
    print("Creating Gemini chat model...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    # Create agent with memory enabled
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=35,
        memory_enabled=True
    )

    print("\n===== Interactive MCP Chat =====")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("================================\n")
    
    try:
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation...")
                break
            
            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation history cleared.")
                continue
            
            print("\nAssistant: ", end="", flush=True)

            try:
                response = await agent.run(user_input)
                print(response)
            except Exception as e:
                print(f"Error: {e}")
    finally:
        if client and client.sessions:
            await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_memory_chat())
