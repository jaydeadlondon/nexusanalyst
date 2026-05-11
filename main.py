import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_gigachat.chat_models import GigaChat
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


class AgentState(TypedDict):
    question: str
    context_docs: str
    answer: str


llm = GigaChat(credentials=os.getenv("GIGACHAT_CREDENTIALS"), verify_ssl_certs=False)


def doc_specialist_node(state: AgentState):
    """Узел-документовед: ищет информацию в векторной базе."""
    print("--- Шаг 1: Поиск по документам ---")
    query = state["question"]

    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    return {"context_docs": context}


def analyst_node(state: AgentState):
    """Узел-аналитик: пишет ответ на основе найденного контекста."""
    print("--- Шаг 2: Формирование ответа аналитиком ---")

    prompt = f"""Ты — аналитик системы NexusAnalyst. Твоя задача — ответить на вопрос, используя ТОЛЬКО предоставленные документы.
    
    ДОКУМЕНТЫ:
    {state['context_docs']}
    
    ВОПРОС:
    {state['question']}
    
    Если в документах нет ответа, честно скажи: 'В предоставленных материалах информации не найдено'.
    Ответ напиши профессиональным языком."""

    response = llm.invoke(prompt)
    return {"answer": response.content}


builder = StateGraph(AgentState)
builder.add_node("doc_specialist", doc_specialist_node)
builder.add_node("analyst", analyst_node)

builder.set_entry_point("doc_specialist")
builder.add_edge("doc_specialist", "analyst")
builder.add_edge("analyst", END)

memory = MemorySaver()
app = builder.compile(checkpointer=memory)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "session_pdf_1"}}

    print("=== NexusAnalyst: Чат по вашему PDF ===")

    while True:
        user_input = input("\nВаш вопрос по документу: ")
        if user_input.lower() in ["exit", "quit", "выход"]:
            break

        initial_state = {"question": user_input, "context_docs": "", "answer": ""}

        result = app.invoke(initial_state, config=config)

        print("\n[ОТВЕТ]:")
        print(result["answer"])
