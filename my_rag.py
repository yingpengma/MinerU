import os
import json
import logging
import sys
from dotenv import load_dotenv

# LlamaIndex 核心组件
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings
)
# LlamaIndex 模型组件
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- 准备工作: 加载环境变量 ---
# 这会从同级目录下的 .env 文件中加载所有配置
load_dotenv()
print("成功从 .env 文件加载环境变量。")

# 设置日志记录，以便看到详细的输出
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def run_mineru_rag_pipeline():
    """
    一个完整的、端到端的脚本，用于运行基于 mineru 输出的 RAG 系统。
    所有密钥和 URL 均从 .env 文件加载。
    嵌入模型使用云服务 (阿里云 DashScope)。
    """
    print("\n--- STEP 1: 配置模型 (从环境变量加载) ---")
    
    # --- 配置 LLM (问答模型) ---
    llm_api_base = os.getenv("LLM_API_BASE")
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_model_name = os.getenv("LLM_MODEL_NAME")

    if not all([llm_api_base, llm_api_key, llm_model_name]):
        print("错误：请确保 .env 文件中已设置 LLM_API_BASE, LLM_API_KEY, 和 LLM_MODEL_NAME。")
        return

    llm = OpenAI(
        model=llm_model_name,
        api_base=llm_api_base,
        api_key=llm_api_key
    )
    print(f"LLM 已配置为使用自定义接口: {llm_api_base}")

    # --- 配置 Embedding (嵌入模型) ---
    embed_api_base = os.getenv("EMBED_API_BASE")
    embed_api_key = os.getenv("EMBED_API_KEY")
    embed_model_name = os.getenv("EMBED_MODEL_NAME")
    
    if not all([embed_api_base, embed_api_key, embed_model_name]):
        print("错误：请确保 .env 文件中已设置 EMBED_API_BASE, EMBED_API_KEY, 和 EMBED_MODEL_NAME。")
        return

    # 注意：我们使用 OpenAIEmbedding 类，因为它能与任何 OpenAI 兼容的 API 对话
    embed_model = OpenAIEmbedding(
        model=embed_model_name,
        api_base=embed_api_base,
        api_key=embed_api_key,
    )
    print(f"嵌入模型已配置为使用云服务接口: {embed_api_base}")

    # --- 应用全局设置 ---
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print("\n--- STEP 2: 加载并处理您的 JSON 数据 ---")

    json_path = "output/2408.03314v1/auto/2408.03314v1_content_list.json"
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：JSON 文件未找到，请确认路径 '{json_path}' 是否正确。")
        return

    documents = []
    for i, item in enumerate(data):
        if item.get("type") == "text" and item.get("text", "").strip():
            doc = Document(
                text=item["text"],
                metadata={
                    "page": item.get("page_idx", "N/A"),
                    "type": item.get("type"),
                    "level": item.get("text_level", 0),
                    "chunk_id": f"chunk_{i}"
                }
            )
            documents.append(doc)

    if not documents:
        print("错误：从 JSON 文件中没有加载到任何有效的文本块。")
        return
        
    print(f"成功加载并转换了 {len(documents)} 个文本块作为 Document 对象。")

    print("\n--- STEP 3: 创建索引 (Indexing) ---")
    print("正在调用云服务对所有文本块进行嵌入操作，速度将远快于本地。")
    
    index = VectorStoreIndex.from_documents(documents)
    
    print("索引创建完成！")

    print("\n--- STEP 4: 创建查询引擎并提问 ---")
    
    query_engine = index.as_query_engine(similarity_top_k=3)

    query = "What is the 'compute-optimal' scaling strategy? Explain it simply."
    print(f"\n正在执行查询: {query}")
    
    response = query_engine.query(query)

    print("\n--- 最终答案 ---")
    print(str(response))

    print("\n--- 答案来源 (Source Nodes) ---")
    print("以下是 LLM 用来生成答案的原始文本块：")
    for i, node in enumerate(response.source_nodes):
        print(f"\n[来源 {i+1} | 相似度: {node.score:.4f} | 页码: {node.metadata.get('page')}]")
        print("-----------------------------------------------------")
        print(node.get_content().strip())
        print("-----------------------------------------------------")

if __name__ == "__main__":
    run_mineru_rag_pipeline()