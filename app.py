import streamlit as st
import json
import time
import os
import base64
from datetime import datetime
from rag_core import initialize_rag_system, get_reference_map
from streamlit_pdf_viewer import pdf_viewer

# 为实现可追溯性而新增的 import
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload

# --- 新增: 可复用的溯源详情显示函数 ---
def render_trace_expander(event_pairs, pdf_bytes, message_index):
    """根据 event_pairs 在 Streamlit 中渲染可展开的溯源详情。"""
    
    with st.expander("🔍 点击查看本次查询的详细溯源流程", expanded=False):
        if not event_pairs:
            st.warning("未能捕获到任何处理事件。")
            return

        # --- Helper to get duration and format it ---
        def get_duration(etype):
            pair = next((p for p in event_pairs if p[0].event_type == etype), None)
            if pair:
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
            st.text_area("输入文本", value=query_text, height=100, disabled=True, key=f"embed_in_{message_index}")
            st.text_area("输出向量 (仅显示前5个维度)", value=str(embedding_vector[:5])+"...", height=100, disabled=True, key=f"embed_out_{message_index}")

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
                        page_num = node.metadata.get("page")
                        page_info = f" | 原始页码: {page_num}" if page_num is not None else ""
                        st.markdown(f"**文本块 {node_idx+1} (相似度: {node.score:.4f}{page_info})**")
                        
                        if page_num is not None and pdf_bytes:
                            button_key = f"scroll_btn_{message_index}_{node.node_id}"
                            if st.button(f"📜 查看原文第 {page_num} 页", key=button_key):
                                st.session_state.scroll_to_page = page_num + 1
                                st.rerun()

                        st.text_area(f"内容 (Chunk ID: {node.metadata.get('chunk_id')})", value=node.get_content(), height=120, disabled=True, key=f"retrieved_{message_index}_{node_idx}")

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
                    st.text_area("发送给大模型的最终提示词 (Full Prompt)", value=full_prompt_str, height=400, disabled=True, key=f"prompt_{message_index}")
                
                if response_payload:
                    raw_response_str = str(getattr(response_payload, 'raw', response_payload))
                    st.text_area("从大模型收到的原始回复 (Raw Response)", value=raw_response_str, height=200, disabled=True, key=f"response_{message_index}")


# --- 页面配置 ---
st.set_page_config(
    page_title="Mineru RAG System",
    page_icon="⛏️",
    layout="wide"
)

st.title("⛏️ Mineru RAG 系统")
st.write("一个完全透明的交互式问答系统。您将看到从提问到回答的每一个内部步骤。")

# --- 新增: 定义PDF文件路径和加载函数 ---
PDF_PATH = "output/demo2/auto/demo2_origin.pdf"

@st.cache_data
def get_pdf_bytes(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None

pdf_bytes = get_pdf_bytes(PDF_PATH)

# --- 新增: 初始化 session state 用于PDF滚动 ---
if "scroll_to_page" not in st.session_state:
    st.session_state.scroll_to_page = None

# --- STEP 1: 加载并缓存底层的、稳定的 vector_store ---
with st.spinner("正在初始化系统，加载模型和索引... 请稍候..."):
    vector_store = initialize_rag_system()
    reference_map = get_reference_map()

if vector_store is None:
    st.error("系统初始化失败，请检查后台日志。")
    st.stop()

# --- 新增: 创建双栏布局 ---
main_col, pdf_col = st.columns([6, 4])

# --- 在右侧栏显示PDF ---
with pdf_col:
    if pdf_bytes:
        st.subheader("📄 文档预览")
        pdf_viewer(
            input=pdf_bytes, 
            width=700, 
            height=800, 
            scroll_to_page=st.session_state.scroll_to_page
        )
        # 查看后重置，防止后续自动滚动
        if st.session_state.scroll_to_page is not None:
            st.session_state.scroll_to_page = None
    else:
        st.warning("未能找到用于预览的PDF文件。")


# --- 将所有聊天相关UI放入左侧栏 ---
with main_col:
    # --- 聊天界面 ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "您好！我已经学习了您的文档，请问有什么可以帮您的？"}]

    # --- 修改: 显示完整的聊天记录和溯源信息 ---
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            # 如果是助手消息且包含溯源数据，则渲染它
            if msg["role"] == "assistant" and "trace_events" in msg:
                render_trace_expander(msg["trace_events"], pdf_bytes, idx)

    if prompt := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # --- 修改: 提问后立刻刷新，以显示新问题 ---
        st.rerun()

# --- 修改: 将查询逻辑移到主脚本流程中 ---
# 检查最后一个消息是否是用户消息且没有被处理过
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and "trace_events" not in st.session_state.messages[-1]:
    
    prompt = st.session_state.messages[-1]["content"]

    with main_col:
        with st.chat_message("assistant"):
            response_container = st.empty()
            response_container.write("思考中...")

            # a. 为本次查询创建全新的追踪器
            llama_debug = LlamaDebugHandler()
            callback_manager = CallbackManager([llama_debug])
            
            # b. 将追踪器设置在全局 Settings 对象上
            Settings.callback_manager = callback_manager
            
            # c. 从缓存的 vector_store 动态创建 index
            index = VectorStoreIndex.from_vector_store(vector_store)
            
            # d. 创建查询引擎
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

            # --- 修改: 将溯源数据存入 session_state ---
            event_pairs = llama_debug.get_event_pairs()
            
            assistant_message = {
                "role": "assistant",
                "content": response.response,
                "trace_events": event_pairs
            }
            st.session_state.messages.append(assistant_message)
            
            # 最后，重置全局回调管理器
            Settings.callback_manager = CallbackManager([])
            
            # --- 修改: 处理完后再次刷新以显示完整结果 ---
            st.rerun()
