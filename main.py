import os
from typing import TypedDict
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
    is_useful: str


def router_node(state: AgentState):
    """Начальный выбор пути."""
    print("--- 🤖 РОУТЕР: Выбираю начальный путь... ---")
    prompt = f"Вопрос: {state['question']}\nКасается ли вопрос темы документов в PDF? Ответь только PDF или WEB."
    res = llm.invoke(prompt).content.strip().upper()
    route = "PDF" if "PDF" in res else "WEB"
    return {"is_useful": route}


def rag_node(state: AgentState):
    """Поиск в PDF."""
    print("--- 📄 RAG: Извлекаю данные из PDF... ---")
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([d.page_content for d in docs])
    return {"pdf_context": context}


def grade_documents_node(state: AgentState):
    """Проверка качества найденных данных."""
    print("--- ⚖️ ИНСПЕКТОР: Проверяю релевантность PDF... ---")

    prompt = f"""Ты — инспектор качества. Твоя задача: оценить, содержат ли найденные документы ответ на вопрос.
    ВОПРОС: {state['question']}
    ДОКУМЕНТЫ: {state['pdf_context']}
    
    Если в документах есть хотя бы частичный ответ, напиши YES.
    Если документы бесполезны для отве на этот вопрос, напиши NO.
    Ответь ТОЛЬКО одним словом (YES или NO)."""

    res = llm.invoke(prompt).content.strip().upper()
    score = "YES" if "YES" in res else "NO"
    return {"is_useful": score}


def web_node(state: AgentState):
    """Поиск в интернете (вызывается роутером или если инспектор сказал NO)."""
    print("--- 🌐 WEB: Иду за данными в интернет... ---")
    results = web_search.run(state["question"])
    return {"web_context": results}


def synthesis_node(state: AgentState):
    """Итоговый синтез."""
    print("--- 📝 СИНТЕЗ: Формирую финальный отчет... ---")
    context = f"PDF DATA: {state.get('pdf_context', '')}\nWEB DATA: {state.get('web_context', '')}"

    prompt = f"Вопрос: {state['question']}\nДанные: {context}\nДай точный ответ."
    res = llm.invoke(prompt)
    return {"final_answer": res.content}


workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("rag", rag_node)
workflow.add_node("grader", grade_documents_node)
workflow.add_node("web", web_node)
workflow.add_node("synthesis", synthesis_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router", lambda x: x["is_useful"], {"PDF": "rag", "WEB": "web"}
)

workflow.add_edge("rag", "grader")

workflow.add_conditional_edges(
    "grader", lambda x: x["is_useful"], {"YES": "synthesis", "NO": "web"}
)

workflow.add_edge("web", "synthesis")
workflow.add_edge("synthesis", END)

app = workflow.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "boss_level"}}
    print("=== NexusAnalyst: Autonomous Mode ===")
    while True:
        q = input("\nВопрос: ")
        if q.lower() in ["exit", "quit"]:
            break
        res = app.invoke(
            {"question": q, "pdf_context": "", "web_context": "", "is_useful": ""},
            config=config,
        )
        print(f"\n[NexusAnalyst]: {res['final_answer']}")
