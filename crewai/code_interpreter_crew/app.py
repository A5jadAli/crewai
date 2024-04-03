from crewai import Agent, Task, Crew, Process
from interpreter import interpreter
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# Configuration and Tools
llm = ChatOpenAI(model="gpt-4-turbo-preview")
interpreter.auto_run = True
interpreter.llm.model = "openai/gpt-4-turbo-preview"

class CLITool:
    @tool("Executor")
    def execute_cli_command(command: str):
        """Create and Execute code using Open Interpreter."""
        result = interpreter.chat(command)
        return result

# Creating an Agent for CLI tasks
cli_agent = Agent(
    role='Software Engineer',
    goal='Always use Executor Tool. Ability to perform CLI operations, write programs and execute using Exector Tool',
    backstory='Expert in command line operations, creating and executing code.',
    tools=[CLITool.execute_cli_command],
    verbose=True,
    llm=llm 
)

# Defining a Task for CLI operations
cli_task = Task(
    description='Identify the OS and then empty my recycle bin',
    agent=cli_agent,
    tools=[CLITool.execute_cli_command]
)

# Creating a Crew with CLI focus
cli_crew = Crew(
    agents=[cli_agent],
    tasks=[cli_task],
    process=Process.sequential,
    manager_llm=llm
)

# Run the Crew

import gradio as gr

def cli_interface(command):
    cli_task.description = command  
    result = cli_crew.kickoff()
    return result

iface = gr.Interface(
    fn=cli_interface, 
    inputs=gr.Textbox(lines=2, placeholder="What action to take?"), 
    outputs="text",
    title="CLI Command Executor",
    description="Execute CLI commands via a natural language interface."
)

iface.launch()