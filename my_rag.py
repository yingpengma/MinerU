import os
import json
import logging
import sys
from dotenv import load_dotenv

# LlamaIndex 核心组件
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext
)
# LlamaIndex 模型组件
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
# 持久化存储相关
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- 准备工作: 加载环境变量 ---
load_dotenv()
print("成功从 .env 文件加载环境变量。")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def run_mineru_rag_pipeline():
    """
    一个完整的、端到端的脚本，用于运行基于 mineru 输出的 RAG 系统。
    1. 预处理原始 JSON，为其添加 chunk_id 并保存为 input.json。
    2. 基于 input.json 创建或加载持久化的向量索引。
    3. 对文档进行查询并返回可追溯的答案。
    """
    
    # --- STEP 1: 配置模型 (从环境变量加载) ---
    # ... (这部分与您之前的代码完全相同，保持不变) ...
    print("\n--- STEP 1: 配置模型 (从环境变量加载) ---")
    
    llm_api_base = os.getenv("LLM_API_BASE")
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_model_name = os.getenv("LLM_MODEL_NAME")
    if not all([llm_api_base, llm_api_key, llm_model_name]):
        print("错误：请确保 .env 文件中已设置 LLM_API_BASE, LLM_API_KEY, 和 LLM_MODEL_NAME。")
        return
    llm = OpenAI(model=llm_model_name, api_base=llm_api_base, api_key=llm_api_key)
    print(f"LLM 已配置为使用自定义接口: {llm_api_base}")

    embed_api_base = os.getenv("EMBED_API_BASE")
    embed_api_key = os.getenv("EMBED_API_KEY")
    embed_model_name = os.getenv("EMBED_MODEL_NAME")
    if not all([embed_api_base, embed_api_key, embed_model_name]):
        print("错误：请确保 .env 文件中已设置 EMBED_API_BASE, EMBED_API_KEY, 和 EMBED_MODEL_NAME。")
        return
    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        model_name=embed_model_name,
        api_base=embed_api_base, 
        api_key=embed_api_key
    )
    print(f"嵌入模型已配置为使用云服务接口: {embed_api_base}")

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.embed_model.embed_batch_size = 10

    # --- NEW STEP 2: 数据预处理 - 为原始 JSON 添加 chunk_id ---
    print("\n--- STEP 2: 数据预处理 - 生成带 chunk_id 的 input.json ---")
    
    original_json_path = "output/demo2/auto/demo2_content_list.json"
    enriched_json_path = "output/demo2/auto/input.json" # 这是我们处理后的输入文件

    # 如果 enriched_json_path 已存在，我们就跳过这一步，避免重复工作
    if not os.path.exists(enriched_json_path):
        print(f"'{enriched_json_path}' 不存在，正在从 '{original_json_path}' 创建...")
        try:
            with open(original_json_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
        except FileNotFoundError:
            print(f"错误：原始 JSON 文件未找到，请确认路径 '{original_json_path}' 是否正确。")
            return

        enriched_data = []
        for i, item in enumerate(original_data):
            # 为每个块（JSON对象）添加一个唯一的、可预测的 chunk_id
            item['chunk_id'] = f"chunk_{i}"
            enriched_data.append(item)
        
        # 将带有 chunk_id 的新数据写入 input.json 文件
        with open(enriched_json_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, indent=2, ensure_ascii=False)
        
        print(f"成功创建带有 chunk_id 的文件，并保存于: '{enriched_json_path}'")
    else:
        print(f"发现已存在的 '{enriched_json_path}'，将直接使用该文件。")

    # --- STEP 3: 加载处理后的数据并创建 LlamaIndex 文档 ---
    print(f"\n--- STEP 3: 从 '{enriched_json_path}' 加载数据并准备 RAG ---")
    
    try:
        with open(enriched_json_path, 'r', encoding='utf-8') as f:
            data_for_rag = json.load(f)
    except FileNotFoundError:
        print(f"错误：处理后的 JSON 文件 '{enriched_json_path}' 未找到。")
        return
        
    # 创建引用地图，用于最后溯源
    reference_map = {item['chunk_id']: item for item in data_for_rag}
    
    documents = []
    for item in data_for_rag:
        if item.get("type") == "text" and item.get("text", "").strip():
            # 我们从处理后的文件中读取 text 和所有元数据
            doc = Document(
                text=item["text"],
                metadata={
                    "page": item.get("page_idx", "N/A"),
                    "type": item.get("type"),
                    "level": item.get("text_level", 0),
                    "chunk_id": item['chunk_id'] # 确保这个 ID 来自文件
                }
            )
            documents.append(doc)
    
    if not documents:
        print("错误：从 JSON 文件中没有加载到任何有效的文本块。")
        return
    print(f"成功加载并转换了 {len(documents)} 个文本块作为 Document 对象。")


    # --- STEP 4: 加载或创建持久化索引 (Indexing) ---
    print("\n--- STEP 4: 加载或创建持久化索引 ---")
    
    persist_dir = "./chroma_db_storage"
    if os.path.exists(persist_dir):
        print(f"发现已存在的索引，从 '{persist_dir}' 加载...")
        db = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = db.get_or_create_collection("mineru_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        print("索引加载成功！")
    else:
        print("未发现本地索引，正在创建新索引...")
        print("正在调用云服务对所有文本块进行嵌入操作...")
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        print("索引创建完成！现在将其保存到磁盘...")
        index.storage_context.persist(persist_dir=persist_dir)
        print(f"索引已成功保存到 '{persist_dir}'。")

    # --- STEP 5: 创建查询引擎并提问 ---
    print("\n--- STEP 5: 创建查询引擎并提问 ---")
    
    query_engine = index.as_query_engine(similarity_top_k=3)
    query = "What is the 'compute-optimal' scaling strategy? Explain it simply."
    print(f"\n正在执行查询: {query}")
    
    response = query_engine.query(query)

    print("\n--- 最终答案 ---")
    print(str(response))

    print("\n--- 答案来源 (Source Nodes) ---")
    print("以下是 LLM 用来生成答案的原始文本块及其完整的 JSON 引用：")
    for i, node in enumerate(response.source_nodes):
        chunk_id = node.metadata.get('chunk_id')
        original_chunk_json = reference_map.get(chunk_id)
        print(f"\n[来源 {i+1} | 相似度: {node.score:.4f} | 页码: {node.metadata.get('page')} | Chunk ID: {chunk_id}]")
        print("-------------------------- 文本内容 --------------------------")
        print(node.get_content().strip())
        print("------------------------ 原始 JSON 引用 ----------------------")
        print(json.dumps(original_chunk_json, indent=2, ensure_ascii=False))
        print("------------------------------------------------------------")

if __name__ == "__main__":
    run_mineru_rag_pipeline()