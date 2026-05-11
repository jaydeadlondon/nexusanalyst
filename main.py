import os
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv

from langchain_gigachat.chat_models import GigaChat
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
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
llm = GigaChat(credentials=os.getenv("GIGACHAT_CREDENTIALS"), verify_ssl_certs=False)
web_search = DuckDuckGoSearchRun()


class AgentState(TypedDict):
    question: str
    pdf_context: str
    web_context: str
    final_answer: str
    route: str


def router_node(state: AgentState):
    """Решает: вопрос касается содержимого PDF или общей информации?"""
    print("--- Роутер: Анализирую источник знаний... ---")
    prompt = f"""Ты — диспетчер. Вопрс пользователя: {state['question']}
    Определи, касается ли вопрос темы документов (аналитика, инструкции, отчеты) или это общий вопрос по миру/новостям.
    Ответь одним словом: PDF или WEB."""

    res = llm.invoke(prompt).content.strip().upper()
    route = "PDF" if "PDF" in res else "WEB"
    return {"route": route}


def rag_node(state: AgentState):
    """Ищет в документах."""
    print("--- RAG: Поиск в локальной базе... ---")
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([d.page_content for d in docs])
    return {"pdf_context": context}


def web_node(state: AgentState):
    """Ищет в интернете."""
    print("--- WEB: Поиск в сети... ---")
    results = web_search.run(state["question"])
    return {"web_context": results}


def synthesis_node(state: AgentState):
    """Объединяет всё и пишет финальный ответ."""
    print("--- Синтез: Формирую ответ... ---")
    pdf_data = state.get("pdf_context", "В документах ничего не найдено.")
    web_data = state.get("web_context", "В интернете поиск не проводился.")

    prompt = f"""Ты — NexusAnalyst. Сформируй ответ на вопрос: {state['question']}
    Используй эти данные из PDF: {pdf_data}
    Используй эти данные из сети: {web_data}
    Дай краткий и точный ответ на русском языке."""

    res = llm.invoke(prompt)
    return {"final_answer": res.content}


builder = StateGraph(AgentState)

builder.add_node("router", router_node)
builder.add_node("rag_node", rag_node)
builder.add_node("web_node", web_node)
builder.add_node("synthesis", synthesis_node)

builder.set_entry_point("router")

builder.add_conditional_edges(
    "router", lambda x: x["route"], {"PDF": "rag_node", "WEB": "web_node"}
)

builder.add_edge("rag_node", "synthesis")
builder.add_edge("web_node", "synthesis")
builder.add_edge("synthesis", END)

memory = MemorySaver()
app = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "hybrid_1"}}

    while True:
        q = input("\nВаш вопрос: ")
        if q.lower() in ["exit", "quit"]:
            break

        result = app.invoke(
            {"question": q, "pdf_context": "", "web_context": ""}, config=config
        )
        print(f"\n[NexusAnalyst]: {result['final_answer']}")
