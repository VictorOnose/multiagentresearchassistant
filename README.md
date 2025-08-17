## Project Title: Multi-Agent Research Assistant

**Description:**

This project aims to develop a multi-agent system using LangGraph and Gemini. The system consists of three core agents:

* **Researcher Agent:** This agent will be responsible for gathering information on a specified topic using a search tool.
* **Writer Agent:** This agent will synthesize the findings from the Researcher Agent into a structured summary, which can be formatted as a markdown report or a JSON object.
* **Critique Agent:** This agent will gather his own information through another model and create his own summary.

After each agent gathers and synthesizes information, the summaries are compared with RAGAS metrics to check the corectness of the provided information.

**Requirements:**

The project requires the use of **LangChain** and **LangGraph**. For the Language Models, we used Ollama LLMS: Gemma3 and Llama3.1.


