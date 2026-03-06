# 终端运行：streamlit run LangChain-RAG-QA.py
import streamlit as st
import tempfile
import os

from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import TextLoader

# 修改1
from langchain_community.embeddings import DashScopeEmbeddings

from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.agents import create_react_agent
from langchain_classic.agents import AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# 修改2
from langchain_community.chat_models import ChatTongyi


# 页面配置
st.set_page_config(page_title="文档问答", layout="wide")
st.title("文档问答")

# ================= API配置 =================
with st.sidebar:
    st.subheader("API配置")

    ai_api_key = st.text_input("DashScope API Key", type="password")

    if ai_api_key:
        os.environ["DASHSCOPE_API_KEY"] = ai_api_key

if not ai_api_key:
    st.warning("请在侧边栏输入DashScope API Key后继续")
    st.stop()
# ==========================================


# 上传TXT文件
uploaded_files = st.sidebar.file_uploader(
    label="上传txt文件",
    type=["txt"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("请先上传TXT文档。")
    st.stop()


# ================= 构建检索器 =================
@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):

    docs = []

    temp_dir = tempfile.TemporaryDirectory(dir=r"/Users/e/Desktop/code-design")

    for file in uploaded_files:

        temp_filepath = os.path.join(temp_dir.name, file.name)

        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())

        loader = TextLoader(temp_filepath, encoding="utf-8")

        docs.extend(loader.load())

    # 文档切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(docs)

    # ===== 修改3：Embedding（DashScope）=====
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2"
    )

    vectordb = Chroma.from_documents(splits, embeddings)

    retriever = vectordb.as_retriever()

    return retriever


retriever = configure_retriever(uploaded_files)
# ==============================================


# 初始化聊天记录
if "messages" not in st.session_state or st.sidebar.button("清空聊天记录"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "您好，我是文档问答助手"}
    ]

# 显示历史消息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ================= 创建检索工具 =================
from langchain_classic.tools.retriever import create_retriever_tool

tool = create_retriever_tool(
    retriever,
    name="文档检索",
    description="用于检索用户提出的问题，并基于检索到的文档内容进行回复。",
)

tools = [tool]
# ===============================================


# 聊天历史
msgs = StreamlitChatMessageHistory()

memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output"
)


# ================= Prompt =================
instructions = """您是一个设计用于查询文档来回答问题的代理。
您可以使用文档检索工具，并基于检索内容来回答问题。
如果您从文档中找不到任何信息用于回答问题，则只需返回“抱歉，这个问题我还不知道。”。
"""

base_prompt_template = """
{instructions}

TOOLS:
------

You have access to the following tools:

{tools}

Tool names:
{tool_names}

To use a tool, please use the format:

Thought: Do I need to use a tool? Yes
Action: the action to take
Action Input: the input
Observation: the result

When you have a response:

Thought: Do I need to use a tool? No
Final Answer: your response


Previous conversation history:
{chat_history}

New input: {input}

{agent_scratchpad}
"""

base_prompt = PromptTemplate.from_template(base_prompt_template)

prompt = base_prompt.partial(instructions=instructions)
# ==========================================


# ================= LLM（千问）=================
llm = ChatTongyi(
    model="qwen-plus",
    temperature=0
)
# ============================================


# 创建Agent
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors="没有从知识库检索到相似内容"
)


# 用户输入
user_query = st.chat_input(placeholder="请开始提问吧！")

if user_query:

    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):

        st_cb = StreamlitCallbackHandler(st.container())

        config = {"callbacks": [st_cb]}

        response = agent_executor.invoke(
            {"input": user_query},
            config=config
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": response["output"]}
        )

        st.write(response["output"])