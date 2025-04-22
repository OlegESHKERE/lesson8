import os
from dotenv import load_dotenv
from pathlib import Path

# Імпорт компонентів LlamaIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def configure_environment() -> None:
    """Завантажує змінні оточення з .env файлу"""
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("Відсутній OPENROUTER_API_KEY у .env файлі")

def initialize_llm() -> OpenRouter:
    """Ініціалізує LLM через OpenRouter API"""
    return OpenRouter(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="openai/gpt-4o-mini",
        max_tokens=2048,
        temperature=0.6,
        additional_headers={
            "HTTP-Referer": "https://my-ai-app.com",
            "X-Title": "Python AI"
        }
    )

def setup_embedding_model() -> HuggingFaceEmbedding:
    """Налаштовує модель для ембеддингів"""
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

def load_and_index_data(data_dir: str = "documents") -> VectorStoreIndex:
    """Завантажує документи та створює індекс"""
    if not Path(data_dir).exists():
        raise FileNotFoundError(f"Директорія {data_dir} не знайдена")
    
    docs = SimpleDirectoryReader(data_dir).load_data()
    return VectorStoreIndex.from_documents(docs)

def main():
    try:
        # Налаштування
        configure_environment()
        
        # Ініціалізація моделей
        llm_instance = initialize_llm()
        embedding_model = setup_embedding_model()
        
        # Конфігурація параметрів (правильний спосіб)
        Settings.llm = llm_instance
        Settings.embed_model = embedding_model
        Settings.chunk_size = 1024
        
        # Завантаження даних та індексація
        vector_index = load_and_index_data()
        query_interface = vector_index.as_query_engine()
        
        # Отримання запиту від користувача
        user_query = input("\nВведіть ваш запит: ")
        if not user_query.strip():
            print("Помилка: Запит не може бути порожнім")
            return
        
        # Виконання запиту та вивід результату
        print("\nОбробка запиту...\n")
        result = query_interface.query(user_query)
        print(f"Відповідь: {result}\n")
        
    except Exception as e:
        print(f"Сталася помилка: {str(e)}")

if __name__ == "__main__":
    print("=== Система аналізу текстів ===")
    main()