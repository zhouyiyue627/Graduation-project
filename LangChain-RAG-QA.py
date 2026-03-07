# streamlit run LangChain-RAG-QA.py

import streamlit as st
import tempfile
import os
import re

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatTongyi


# ================= 页面配置 =================

st.set_page_config(
    page_title="RAG 文档问答",
    layout="wide"
)

st.markdown("""
<style>

/* ===== 侧边栏 ===== */

section[data-testid="stSidebar"]{
padding-top:10px;
}

.sidebar-title{
font-size:22px;
font-weight:700;
margin-top:18px;
margin-bottom:8px;
}

/* ===== 主页面 ===== */

/* 主标题 */
h1{
font-size:42px !important;
font-weight:800 !important;
letter-spacing:1px;
margin-bottom:10px;
}

/* 二级标题 */
h2{
font-size:22px !important;
font-weight:600 !important;
margin-top:22px !important;
}

/* 正文 */
p{
font-size:16px;
line-height:1.8;
}

/* 引用角标 */
.ref{
color:#1f77b4;
font-size:13px;
font-weight:600;
margin-left:3px;
}

/* 引用角标链接 */
.ref a{
color:#1f77b4;
text-decoration:none;
}

</style>
""", unsafe_allow_html=True)


st.title("📚 文档问答系统")


# ================= API =================

with st.sidebar:

    st.markdown("<div class='sidebar-title'>🔑 API配置</div>",unsafe_allow_html=True)

    api_key=st.text_input("API Key",type="password")

    if api_key:
        os.environ["DASHSCOPE_API_KEY"]=api_key

if not api_key:
    st.warning("请输入API Key")
    st.stop()


# ================= 上传 =================

st.sidebar.markdown("<div class='sidebar-title'>📂 上传TXT文件</div>",unsafe_allow_html=True)

uploaded_files=st.sidebar.file_uploader(
    "选择 TXT 文件",
    type=["txt"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("请上传TXT文件")
    st.stop()


# ================= 文档浏览 =================

st.sidebar.markdown("<div class='sidebar-title'>📑 文档浏览</div>",unsafe_allow_html=True)

for f in uploaded_files:

    with st.sidebar.expander(f.name):

        preview=f.read().decode("utf-8")[:500]

        st.write(preview+"...")


# ================= 参数 =================

st.sidebar.markdown("<div class='sidebar-title'>⚙️ Top-K 检索数量</div>",unsafe_allow_html=True)

top_k=st.sidebar.slider(
    "检索文档数量",
    1,10,5
)

st.sidebar.markdown("<div class='sidebar-title'>🐞 RAG Debug</div>",unsafe_allow_html=True)

debug_mode=st.sidebar.checkbox("显示检索过程")


# ================= 构建向量库 =================

@st.cache_resource
def build_vector_db(uploaded_files):

    docs=[]

    temp_dir=tempfile.TemporaryDirectory()

    for file in uploaded_files:

        path=os.path.join(temp_dir.name,file.name)

        with open(path,"wb") as f:
            f.write(file.getvalue())

        loader=TextLoader(path,encoding="utf-8")

        docs.extend(loader.load())

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits=splitter.split_documents(docs)

    embeddings=DashScopeEmbeddings(
        model="text-embedding-v2"
    )

    vectordb=Chroma.from_documents(
        splits,
        embeddings
    )

    return vectordb


vectordb=build_vector_db(uploaded_files)


# ================= 高亮 =================

def highlight(text,query):

    words=query.split()

    for w in words:

        text=re.sub(
            re.escape(w),
            f"<mark style='background:#ffe066'>{w}</mark>",
            text,
            flags=re.IGNORECASE
        )

    return text

# ================= 引用证据高亮 =================

def highlight_evidence(text, answer):

    # 从回答里提取关键词
    words = re.findall(r'[\u4e00-\u9fa5]{2,}', answer)

    words = list(set(words))[:8]   # 限制数量，避免过多高亮

    for w in words:

        text = re.sub(
            re.escape(w),
            f"<mark style='background:#ffd54f;font-weight:bold'>{w}</mark>",
            text
        )

    return text


# ================= 引用脚标（可点击） =================

def style_refs(answer):

    def replace_ref(match):

        num=match.group(1)

        return f"<sup class='ref'><a href='#ref-{num}'>[{num}]</a></sup>"

    answer=re.sub(
        r"\[(\d+)\]",
        replace_ref,
        answer
    )

    return answer


# ================= 检索 =================

def get_sources(query):

    docs_scores=vectordb.similarity_search_with_score(
        query,
        k=top_k
    )

    results=[]

    for doc,score in docs_scores:

        similarity=1/(1+score)

        similarity=round(similarity,3)

        results.append({

        "content":doc.page_content,

        "source":os.path.basename(
        doc.metadata.get("source","未知")
        ),

        "score":similarity
        })

    results=sorted(results,key=lambda x:x["score"],reverse=True)

    return results


# ================= LLM =================

llm=ChatTongyi(
model="qwen-plus",
temperature=0
)


# ================= 聊天历史 =================

if "messages" not in st.session_state:

    st.session_state.messages=[
    {"role":"assistant","content":"嗨！我是文档问答小助手，随时为你解答文档相关问题～"}
    ]


for msg in st.session_state.messages:

    st.chat_message(msg["role"]).write(msg["content"])


# ================= 用户输入 =================

query=st.chat_input("请输入问题")

if query:

    st.session_state.messages.append(
    {"role":"user","content":query}
    )

    st.chat_message("user").write(query)


    with st.chat_message("assistant"):

        sources=get_sources(query)

        context=""

        for i,src in enumerate(sources,start=1):

            context+=f"\n[{i}] {src['content']}\n"


        rag_prompt=f"""
请根据文档回答问题，并在引用处标注数字。

问题:
{query}

文档:
{context}

引用示例:
[1][2]
"""

        answer=llm.invoke(rag_prompt).content


        refs=re.findall(r"\[(\d+)\]",answer)

        used=sorted(set(map(int,refs)))

        mapping={old:i+1 for i,old in enumerate(used)}

        for old, new in mapping.items():
            answer = answer.replace(f"[{old}]", f"[{new}]")

        # 删除回答里的“引用来源”
        answer = re.sub(r"引用来源[:：].*", "", answer)

        # 删除单独一行的引用编号，例如：[1] [2]
        answer = re.sub(r"\n?\s*(\[\d+\]\s*)+\s*$", "", answer)

        styled_answer=style_refs(answer)

        st.markdown(styled_answer,unsafe_allow_html=True)

        # ===== 引用出处 =====

        if used:

            st.markdown("## 📎 引用出处")

            for old, new in mapping.items():
                src = sources[old - 1]

                st.markdown(
                    f"<div id='ref-{new}'></div>",
                    unsafe_allow_html=True
                )

                with st.expander(f"[{new}] 📄 {src['source']}"):
                    evidence = highlight_evidence(
                        src["content"],
                        answer
                    )

                    st.markdown(
                        f"""
                        <div style="
                        background:#f8f9fa;
                        padding:14px;
                        border-radius:8px;
                        line-height:1.8;
                        font-size:15px;
                        ">
                        {evidence}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


        # ===== 检索结果 =====

        st.markdown("## 🔎 检索结果")

        for i,src in enumerate(sources,start=1):

            score=round(src["score"]*100,1)

            with st.expander(f"{i}. {src['source']} ｜ 相似度 {score}%"):

                text=highlight(src["content"],query)

                st.markdown(text,unsafe_allow_html=True)

                st.progress(src["score"])