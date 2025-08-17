## Project Title: Multi-Agent Research Assistant

**Description:**

This project aims to develop a multi-agent system using LangGraph and Gemini. The system will consists of three core agents:

* **Researcher Agent:** This agent will be responsible for gathering information on a specified topic using a search tool.
* **Writer Agent:** This agent will synthesize the findings from the Researcher Agent into a structured summary, which can be formatted as a markdown report or a JSON object.
* **Critique Agent:** This agent will gather his own information through another model and create his own summary, 

Additional components will be integrated throughout the development process.

**Requirements:**

The project requires the use of **LangChain** and **LangGraph**. For the Language Model, you can choose a **Gemini model** from the available options. Using **Ollama** for the LLM will earn bonus points.

**Main Steps for Guidance:**

1.  **Architectural Design:** Begin by conceptualizing the application's flow and identifying the necessary nodes for the graph.
2.  **State Object Creation:** Define the state object that will manage information across the agents.
3.  **Function Development:** Write the individual functions that will interact with your chosen initialized LLM.
4.  **Graph Construction:** Connect these functions to form the LangGraph, with each function serving as a distinct node.
