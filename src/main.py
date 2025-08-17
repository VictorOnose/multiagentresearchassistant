import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import display, Image
from langchain_ollama import ChatOllama

from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference

class State(TypedDict):
    messages: Annotated[List[AIMessage], "The messages in the conversation"]
    topic: str
    information: str
    summary: str
    summary_critique: str

def supervisor(state: State) -> State:
    print("Supervisor is overseeing the process...")
    if state['messages'] and isinstance(state['messages'][0], HumanMessage):
        user_message = state['messages'][0].content
        topic = user_message  
    else:
        topic = state.get('topic', 'Unknown topic')
    
    print(f"Topic identified: {topic}")
    return {
        **state,
        "topic": topic,
        "messages": state['messages'] + [AIMessage(content="Supervisor is overseeing the process.")],
    }

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0.4
)

llm_critique = ChatOllama(
    model="llama3.1:8b",
    temperature=0.1
)

supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the Supervisor. Your task is to oversee the Researcher and Writer. You will ensure that the Researcher gathers the necessary information and that the Writer summarizes it effectively."),
    ("human", "I need you to give me details about a {topic} topic."),
])

research_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the Researcher. Your task is to gather information on a specific topic: {topic}. Provide comprehensive information about this topic."),
    ("human", "Gather information on {topic}."),
])

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the Writer. Your task is to summarize the information provided. You will return the summary through a clear and structured format."),
    ("human", "Summarize the following information: {information}. Please provide a concise and structured summary. The summary should be clear and to the point."),
])

critique_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the Critic. Your task is to independently research the topic '{topic}' and create your own comprehensive summary. This will serve as a reference to evaluate other summaries."),
    ("human", "Research and summarize the topic: {topic}. Provide a comprehensive, well-structured summary that covers all important aspects of this topic."),
])

def research_topic(state: State) -> State:
    print(f"Researching topic: {state['topic']}")
    
    response = llm.invoke(
        research_prompt.format(topic=state['topic'])
    )
    
    print("Research completed. Information gathered:")
    print(f"First 200 chars: {response.content[:200]}...")
    
    return {
        **state,
        "information": response.content,
        "messages": state['messages'] + [AIMessage(content=f"Research completed on: {state['topic']}")],
    }

def write_summary(state: State) -> State:
    print(f"Summarizing information on: {state['topic']}")
    
    response = llm.invoke(
        summary_prompt.format(information=state['information'])
    )
    
    print("\nSummarized information:")
    print(response.content)
    
    return {
        **state,
        "messages": state['messages'] + [AIMessage(content=response.content)],
        "summary": response.content,
    }

def critique_summary(state: State) -> State:
    print(f"Critic independently researching and summarizing topic: {state['topic']}")
    
    response = llm_critique.invoke(
        critique_prompt.format(topic=state['topic'])
    )
    
    print("\nCritic's independent summary:")
    print(response.content)
    
    return {
        **state,
        "messages": state['messages'] + [AIMessage(content=response.content)],
        "summary_critique": response.content,
    }

def decision_function(state: State) -> str:
    print("\nEvaluating summaries with RAGAS...")
    
    try:
        context_precision = NonLLMContextPrecisionWithReference()
        sample = SingleTurnSample(
            retrieved_contexts=[state['summary']],
            reference_contexts=[state['summary_critique']]
        )
        
        score = context_precision.single_turn_score(sample)
        print(f"Context Precision Score: {score}")
        
        if score > 0.5:
            print("Score > 0.5: Ending workflow")
            return END
        else:
            print("Score <= 0.5: Going back to supervisor for refinement")
            return "supervisor"
            
    except Exception as e:
        print(f"Error in RAGAS evaluation: {e}")
        print("Defaulting to END")
        return END

workflow = StateGraph(State)

workflow.add_node("supervisor", supervisor)
workflow.add_node("research_topic", research_topic)
workflow.add_node("write_summary", write_summary)
workflow.add_node("critique_summary", critique_summary)
workflow.set_entry_point("supervisor")

workflow.add_edge("supervisor", "research_topic")
workflow.add_edge("research_topic", "write_summary")
workflow.add_edge("write_summary", "critique_summary")
workflow.add_conditional_edges("critique_summary", decision_function)

app = workflow.compile()

try:
    graph_image = app.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )
    with open("workflow_graph.png", "wb") as f:
        f.write(graph_image)
    print("Workflow graph saved as 'workflow_graph.png' in current directory")
except Exception as e:
    print(f"Could not save graph: {e}")

def run_research(user_request: str):
    print(f"Initial Request: {user_request}\n")
    state = {
        "messages": [HumanMessage(content=user_request)],
        "topic": "",
        "information": "",
        "summary": "",
        "summary_critique": "",
    }

    for output in app.stream(state):
        pass
    
    return state

if __name__ == "__main__":
    user_request = "Find out about the impact of climate change on polar bears."
    run_research(user_request)