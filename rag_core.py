import os
import json
import logging
import sys
from dotenv import load_dotenv

# 所有 import 保持不变
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
# 增加 streamlit 的 import
import streamlit as st

# --- 准备工作: 加载环境变量 ---
load_dotenv()
print("成功从 .env 文件加载环境变量。")

# 配置日志
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# 使用 Streamlit 的缓存装饰器，这是关键！
@st.cache_resource
def initialize_rag_system():
    """
    这个函数负责所有的一次性初始化工作。
    它确保向量数据库已填充数据，并返回一个稳定的 vector_store 对象供后续使用。
    """
    print("--- 正在执行一次性初始化... ---")

    # --- STEP 1: 配置全局模型 ---
    # ... (Settings configuration remains the same) ...
    llm_api_base = os.getenv("LLM_API_BASE")
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_model_name = os.getenv("LLM_MODEL_NAME")
    llm = OpenAI(model=llm_model_name, api_base=llm_api_base, api_key=llm_api_key)
    
    embed_api_base = os.getenv("EMBED_API_BASE")
    embed_api_key = os.getenv("EMBED_API_KEY")
    embed_model_name = os.getenv("EMBED_MODEL_NAME")
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", model_name=embed_model_name, api_base=embed_api_base, api_key=embed_api_key)

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.embed_model.embed_batch_size = 10
    print("模型配置完成。")

    # --- STEP 2: 准备向量存储 ---
    persist_dir = "./chroma_db_storage"
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection("mineru_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # --- STEP 3: 检查并按需填充数据库 ---
    if chroma_collection.count() == 0:
        print("向量存储为空，需要创建新索引...")
        original_json_path = "output/demo2/auto/demo2_content_list.json"
        enriched_json_path = "output/demo2/auto/input.json"
        
        if not os.path.exists(enriched_json_path):
            try:
                with open(original_json_path, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
            except FileNotFoundError:
                print(f"错误: 原始 JSON 文件 '{original_json_path}' 未找到。")
                st.error(f"错误: 原始 JSON 文件 '{original_json_path}' 未找到。")
                return None
            enriched_data = [{'chunk_id': f"chunk_{i}", **item} for i, item in enumerate(original_data)]
            with open(enriched_json_path, 'w', encoding='utf-8') as f:
                json.dump(enriched_data, f, indent=2, ensure_ascii=False)

        with open(enriched_json_path, 'r', encoding='utf-8') as f:
            data_for_rag = json.load(f)
        
        documents = [Document(text=item["text"], metadata={"page": item.get("page_idx"), "chunk_id": item['chunk_id']}) for item in data_for_rag if item.get("type") == "text" and item.get("text", "").strip()]
        
        # 这个过程会通过 vector_store 将数据写入 ChromaDB
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
        print("索引创建完成并存入 ChromaDB。")
    else:
        print(f"发现已存在的索引({chroma_collection.count()}条)，跳过创建。")

    # --- 最终返回: 只返回稳定的 vector_store 对象 ---
    print("--- 初始化完成，向量存储已准备就绪 ---")
    return vector_store

def get_reference_map():
    """一个辅助函数，用于加载溯源用的 JSON 数据。"""
    enriched_json_path = "output/demo2/auto/input.json"
    if not os.path.exists(enriched_json_path):
        # 在后端返回错误信息或空字典比在UI层面更合适
        print(f"警告: 溯源文件 '{enriched_json_path}' 未找到。")
        return {}
    with open(enriched_json_path, 'r', encoding='utf-8') as f:
        data_for_rag = json.load(f)
    return {item['chunk_id']: item for item in data_for_rag} 