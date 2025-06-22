import streamlit as st
import json
import time
from datetime import datetime
from rag_core import initialize_rag_system, get_reference_map

# 为实现可追溯性而新增的 import
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload

# --- 页面配置 ---
st.set_page_config(
    page_title="Mineru RAG System (透明版)",
    page_icon="⛏️",
    layout="wide"
)

st.title("⛏️ Mineru RAG 系统 (透明版)")
st.write("一个完全透明的交互式问答系统。您将看到从提问到回答的每一个内部步骤。")

# --- STEP 1: 加载并缓存底层的、稳定的 vector_store ---
with st.spinner("正在初始化系统，加载模型和索引... 请稍候..."):
    vector_store = initialize_rag_system()
    reference_map = get_reference_map()

if vector_store is None:
    st.error("系统初始化失败，请检查后台日志。")
    st.stop()

# --- 聊天界面 ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！我已经学习了您的文档，请问有什么可以帮您的？"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()
        
        # --- 最终修正: 遵循最新的 llama-index 模式 ---
        
        # a. 为本次查询创建全新的追踪器
        llama_debug = LlamaDebugHandler()
        callback_manager = CallbackManager([llama_debug])
        
        # b. 将追踪器设置在全局 Settings 对象上
        Settings.callback_manager = callback_manager
        
        # c. 从缓存的 vector_store 动态创建 index，它会自动继承全局 Settings
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        # d. 从新鲜的 index 创建查询引擎
        query_engine = index.as_query_engine(similarity_top_k=3)
        
        with st.status("正在处理您的问题...", expanded=False) as status:
            status.write("1. 分析问题并调用 Embedding 模型...")
            response = query_engine.query(prompt)
            status.write("2. 在 ChromaDB 中执行向量检索...")
            time.sleep(0.2)
            status.write("3. 将检索结果与问题整合成提示词 (Prompt)...")
            time.sleep(0.2)
            status.write("4. 调用大语言模型 (LLM) 生成答案...")
            time.sleep(0.2)
            status.update(label="答案生成完毕！", state="complete")

        response_container.write(response.response)

        # --- 全新: 显示详细、可用的可追溯流程 ---
        with st.expander("🔍 点击查看本次查询的详细溯源流程", expanded=True):
            event_pairs = llama_debug.get_event_pairs()
            if not event_pairs:
                st.warning("未能捕获到任何处理事件。")
            else:
                # --- NEW: Helper to get duration and format it ---
                def get_duration(etype):
                    pair = next((p for p in event_pairs if p[0].event_type == etype), None)
                    if pair:
                        # CRITICAL FIX 2: Use strptime for the custom timestamp format
                        time_format = "%m/%d/%Y, %H:%M:%S.%f"
                        start_time = datetime.strptime(pair[0].time, time_format)
                        end_time = datetime.strptime(pair[1].time, time_format)
                        return (end_time - start_time).total_seconds()
                    return None
                
                def get_duration_str(etype):
                    duration = get_duration(etype)
                    if duration is not None:
                        return f" - 耗时: {duration:.4f} 秒"
                    return ""

                query_duration = get_duration(CBEventType.QUERY)
                if query_duration is not None:
                    st.info(f"**总查询耗时: {query_duration:.4f} 秒**")

                # 步骤 1: 问题向量化
                embedding_events = [p for p in event_pairs if p[0].event_type == CBEventType.EMBEDDING]
                if embedding_events:
                    st.markdown("---")
                    st.subheader(f"第一步: 问题向量化 (Embedding){get_duration_str(CBEventType.EMBEDDING)}")
                    st.markdown("系统调用 Embedding 模型，将您的文本问题转换成机器可以理解的数字列表（向量）。")
                    query_text = embedding_events[0][1].payload.get(EventPayload.CHUNKS, ["未知输入"])[0]
                    embedding_vector = embedding_events[0][1].payload.get(EventPayload.EMBEDDINGS, [[]])[0]
                    st.text_area("输入文本", value=query_text, height=100, disabled=True, key="embed_in")
                    st.text_area("输出向量 (仅显示前5个维度)", value=str(embedding_vector[:5])+"...", height=100, disabled=True, key="embed_out")

                # 步骤 2: 向量检索
                retrieval_events = [p for p in event_pairs if p[0].event_type == CBEventType.RETRIEVE]
                if retrieval_events:
                    st.markdown("---")
                    st.subheader(f"第二步: 向量检索 (Retrieval){get_duration_str(CBEventType.RETRIEVE)}")
                    st.markdown('系统使用"问题向量"，在 ChromaDB 中搜索语义最相关的文本块。')
                    retrieved_nodes = retrieval_events[0][1].payload.get(EventPayload.NODES, [])
                    if retrieved_nodes:
                        st.success(f"在数据库中成功检索到 {len(retrieved_nodes)} 个相关文本块。")
                        for node_idx, node in enumerate(retrieved_nodes):
                            with st.container(border=True):
                                st.markdown(f"**文本块 {node_idx+1} (相似度: {node.score:.4f})**")
                                st.text_area(f"内容 (Chunk ID: {node.metadata.get('chunk_id')})", value=node.get_content(), height=120, disabled=True, key=f"retrieved_{node_idx}")

                # 步骤 3: 答案生成
                synthesize_events = [p for p in event_pairs if p[0].event_type == CBEventType.SYNTHESIZE]
                if synthesize_events:
                    st.markdown("---")
                    st.subheader(f"第三步: 答案生成 (LLM Synthesize){get_duration_str(CBEventType.SYNTHESIZE)}")
                    st.markdown('系统将检索到的文本块作为"上下文"，连同原始问题组合成最终提示词 (Prompt)，发送给大语言模型 (LLM) 生成答案。')
                    
                    llm_duration = get_duration(CBEventType.LLM)
                    if llm_duration is not None:
                        st.caption(f"其中，大语言模型调用 (LLM) 耗时 {llm_duration:.4f} 秒")

                    llm_events = [p for p in event_pairs if p[0].event_type == CBEventType.LLM]
                    if llm_events:
                        prompt_payload = llm_events[0][0].payload.get(EventPayload.PROMPT) or llm_events[0][0].payload.get(EventPayload.MESSAGES)
                        response_payload = llm_events[0][1].payload.get(EventPayload.RESPONSE) or llm_events[0][1].payload.get(EventPayload.LLM_RESPONSE)
                        
                        if prompt_payload:
                            full_prompt_str = "\n\n---\n\n".join([str(p) for p in prompt_payload]) if isinstance(prompt_payload, list) else str(prompt_payload)
                            st.text_area("发送给大模型的最终提示词 (Full Prompt)", value=full_prompt_str, height=400, disabled=True, key="prompt")
                        
                        if response_payload:
                            raw_response_str = str(getattr(response_payload, 'raw', response_payload))
                            st.text_area("从大模型收到的原始回复 (Raw Response)", value=raw_response_str, height=200, disabled=True, key="response")

    # 将助手的纯文本回答添加到历史记录
    if "response" in locals() and hasattr(response, 'response'):
        st.session_state.messages.append({"role": "assistant", "content": response.response})
    
    # 最后，重置全局回调管理器
    Settings.callback_manager = CallbackManager([])
