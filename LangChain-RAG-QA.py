# streamlit run LangChain-RAG-QA.py

import streamlit as st
import tempfile
import os
import re
import math
import base64
import jieba
from collections import Counter
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
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

/* 文件类型徽标 */
.file-badge {
    display:inline-block;
    padding:2px 8px;
    border-radius:4px;
    font-size:12px;
    font-weight:700;
    margin-right:6px;
}
.badge-txt  { background:#e8f4f8; color:#1565c0; }
.badge-pdf  { background:#fdecea; color:#c62828; }
.badge-docx { background:#e8f5e9; color:#2e7d32; }
.badge-doc  { background:#f1f8e9; color:#558b2f; }
.badge-img  { background:#fff8e1; color:#f57f17; }
.badge-unknown { background:#f5f5f5; color:#616161; }

/* ===== 文件上传组件汉化 ===== */

/* 隐藏"Drag and drop files here" */
.ewslnz93 {
    display: none !important;
}
/* 用新元素替代显示中文 */
[data-testid="stFileUploaderDropzoneInstructions"] > div::before {
    content: "拖拽文件到此处";
    display: block;
    font-size: 14px;
    color: rgb(49, 51, 63);
    margin-bottom: 4px;
}

/* 隐藏"Limit 200MB per file..." */
.ewslnz94 {
    display: none !important;
}
/* 用新元素替代显示中文 */
[data-testid="stFileUploaderDropzoneInstructions"] > div::after {
    content: "每个文件限制 200MB";
    display: block;
    font-size: 12px;
    color: #888;
    margin-top: 4px;
}

/* 隐藏英文"Browse files" —— 兼容新旧版本 */
[data-testid="stFileUploaderDropzone"] button[data-testid="baseButton-secondary"],
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] {
    color: transparent !important;
    position: relative;
    overflow: hidden;
}
/* 替换为中文"浏览文件" */
[data-testid="stFileUploaderDropzone"] button[data-testid="baseButton-secondary"]::after,
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]::after {
    content: "浏览文件";
    color: rgb(49, 51, 63);
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    white-space: nowrap;
    pointer-events: none;
    font-size: 14px; 
}

</style>
""", unsafe_allow_html=True)

st.title("📚 文档问答系统")

# ================= 文件类型配置 =================

SUPPORTED_TYPES = ["txt", "pdf", "docx", "doc", "png", "jpg", "jpeg"]

FILE_ICONS = {
    "txt":  "📄",
    "pdf":  "📕",
    "docx": "📘",
    "doc":  "📗",
    "png":  "🖼️",
    "jpg":  "🖼️",
    "jpeg": "🖼️",
}

IMAGE_TYPES = {"png", "jpg", "jpeg"}

# ================= API =================

with st.sidebar:
    st.markdown("<div class='sidebar-title'>🔑 API配置</div>", unsafe_allow_html=True)
    api_key = st.text_input("API Key", type="password")
    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key

if not api_key:
    st.warning("请输入API Key")
    st.stop()

# ================= 上传（支持多格式） =================

st.sidebar.markdown("<div class='sidebar-title'>📂 上传文件</div>", unsafe_allow_html=True)
st.sidebar.caption("支持格式：TXT・PDF・Word・PNG/JPG")

uploaded_files = st.sidebar.file_uploader(
    "选择文件",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("请上传文件（支持 TXT、PDF、Word、图片）")
    st.stop()


# ================= 工具函数 =================

def get_ext(filename: str) -> str:
    return Path(filename).suffix.lstrip(".").lower()


def file_badge(ext: str) -> str:
    """生成文件类型徽标 HTML"""
    label = ext.upper()
    css_class = f"badge-{ext}" if ext in ("txt", "pdf", "docx", "doc") else (
        "badge-img" if ext in IMAGE_TYPES else "badge-unknown"
    )
    return f"<span class='file-badge {css_class}'>{label}</span>"


# ================= 侧边栏文档预览（支持多格式） =================

st.sidebar.markdown("<div class='sidebar-title'>📑 文档预览</div>", unsafe_allow_html=True)

for f in uploaded_files:
    ext = get_ext(f.name)
    icon = FILE_ICONS.get(ext, "📎")

    with st.sidebar.expander(f"{icon} {f.name}"):
        if ext == "txt":
            try:
                preview = f.getvalue().decode("utf-8")[:500]
                st.write(preview + ("..." if len(f.getvalue()) > 500 else ""))
            except UnicodeDecodeError:
                st.warning("文件编码不是 UTF-8，无法预览")

        elif ext == "pdf":
            size_kb = len(f.getvalue()) / 1024
            st.caption(f"📕 PDF 文件 · {size_kb:.1f} KB")
            st.info("PDF 内容将在构建向量库时自动提取")

        elif ext in ("docx", "doc"):
            size_kb = len(f.getvalue()) / 1024
            st.caption(f"📘 Word 文件 · {size_kb:.1f} KB")
            st.info("Word 内容将在构建向量库时自动提取")

        elif ext in IMAGE_TYPES:
            st.image(f.getvalue(), use_container_width=True)
            st.caption("🖼️ 图片将通过通义千问VL提取文字内容")

        else:
            st.warning(f"不支持预览此格式：.{ext}")

# ================= 参数 =================

st.sidebar.markdown("<div class='sidebar-title'>⚙️ Top-K 检索数量</div>", unsafe_allow_html=True)

top_k = st.sidebar.slider(
    "检索文档数量",
    1, 10, 5
)

st.sidebar.markdown("<div class='sidebar-title'>🐞 RAG Debug</div>", unsafe_allow_html=True)

debug_mode = st.sidebar.checkbox("显示检索结果")

st.sidebar.markdown("<div class='sidebar-title'>💻 控制台调试输出</div>", unsafe_allow_html=True)
console_output = st.sidebar.checkbox("启用控制台调试输出", value=False)


# ================= 图片 OCR：通义千问VL =================

def extract_text_from_image_qwen(image_bytes: bytes, filename: str, api_key: str) -> str:
    """
    调用通义千问VL模型提取图片中的文字内容。
    使用 DashScope 原生 API（multimodal-generation）。
    """
    import dashscope
    from dashscope import MultiModalConversation

    dashscope.api_key = api_key

    # 将图片转为 base64
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    ext = get_ext(filename)
    mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
    image_data_url = f"data:{mime};base64,{b64}"

    try:
        response = MultiModalConversation.call(
            model="qwen-vl-plus",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": image_data_url},
                        {"text": "请提取并输出这张图片中所有的文字内容，保持原有格式，不要添加任何解释。"}
                    ]
                }
            ]
        )

        if response.status_code == 200:
            return response.output.choices[0].message.content[0]["text"]
        else:
            return f"[图片OCR失败：{response.message}]"

    except Exception as e:
        return f"[图片OCR异常：{str(e)}]"


# ================= 统一文件加载器 =================

def load_single_file(path: str, filename: str, api_key: str) -> list[Document]:
    """
    根据文件扩展名选择对应的加载器，统一返回 List[Document]。
    包含完整错误处理，单文件失败不影响其他文件。
    """
    ext = get_ext(filename)

    try:
        if ext == "txt":
            # 自动尝试 UTF-8，失败后降级为 GBK（常见中文编码）
            try:
                loader = TextLoader(path, encoding="utf-8")
                return loader.load()
            except UnicodeDecodeError:
                loader = TextLoader(path, encoding="gbk")
                return loader.load()

        elif ext == "pdf":
            loader = PyPDFLoader(path)
            docs = loader.load()
            if not docs or all(not d.page_content.strip() for d in docs):
                # PDF 可能是扫描件，提示用户
                return [Document(
                    page_content=f"[{filename}：PDF内容为空，可能是扫描件图片型PDF，建议转换为图片后上传]",
                    metadata={"source": filename}
                )]
            return docs

        elif ext == "docx":
            loader = Docx2txtLoader(path)
            return loader.load()

        elif ext == "doc":
            # .doc 需要 unstructured 库
            try:
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                loader = UnstructuredWordDocumentLoader(path)
                return loader.load()
            except ImportError:
                return [Document(
                    page_content=f"[{filename}：处理 .doc 文件需要安装 unstructured 库：pip install unstructured]",
                    metadata={"source": filename}
                )]

        elif ext in IMAGE_TYPES:
            with open(path, "rb") as f:
                image_bytes = f.read()
            extracted_text = extract_text_from_image_qwen(image_bytes, filename, api_key)
            return [Document(
                page_content=extracted_text,
                metadata={"source": filename, "type": "image_ocr"}
            )]

        else:
            return [Document(
                page_content=f"[{filename}：不支持的文件格式 .{ext}]",
                metadata={"source": filename}
            )]

    except Exception as e:
        # 单文件异常：不中断整体流程，记录错误信息
        error_msg = f"[{filename} 加载失败：{type(e).__name__}: {str(e)}]"
        return [Document(
            page_content=error_msg,
            metadata={"source": filename, "error": True}
        )]


# ================= 构建向量库 =================

@st.cache_resource
def build_vector_db(uploaded_files, _file_hashes, api_key):
    """
    构建向量数据库。
    _file_hashes 用于缓存失效（文件内容或API Key变化时重建）。
    """
    docs = []
    failed_files = []

    temp_dir = tempfile.TemporaryDirectory()

    # 进度条
    progress = st.progress(0, text="正在处理文件...")
    total = len(uploaded_files)

    for i, file in enumerate(uploaded_files):
        ext = get_ext(file.name)
        icon = FILE_ICONS.get(ext, "📎")
        progress.progress(
            (i + 1) / total,
            text=f"正在处理 {icon} {file.name} ({i+1}/{total})"
        )

        path = os.path.join(temp_dir.name, file.name)
        with open(path, "wb") as f:
            f.write(file.getvalue())

        loaded = load_single_file(path, file.name, api_key)

        # 检查是否加载失败（错误文档）
        for doc in loaded:
            if doc.metadata.get("error"):
                failed_files.append(file.name)
                break

        docs.extend(loaded)

    progress.empty()

    # 显示加载失败的文件
    if failed_files:
        st.warning(f"以下文件加载时出现问题，请检查：{', '.join(failed_files)}")

    if not docs:
        st.error("所有文件均加载失败，无法构建向量库")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = splitter.split_documents(docs)

    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    vectordb = Chroma.from_documents(splits, embeddings)

    return vectordb


# 生成文件哈希值（含 api_key 防止切换key后用旧向量库）
file_hashes = tuple(hash(f.getvalue()) for f in uploaded_files)
vectordb = build_vector_db(uploaded_files, file_hashes, api_key)

# ================= 停用词 =================

STOP_WORDS = {
    # 代词
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
    "自己", "大家", "咱们", "某些", "另外",

    # 指示代词
    "这", "那", "这个", "那个", "这些", "那些", "这样", "那样",
    "此", "彼", "这里", "那里", "这儿", "那儿",

    # 疑问词（保留"如何"、"怎么"、"为什么"等有查询意义的词）
    "什么", "谁", "哪", "哪个", "哪些", "哪里", "哪儿",

    # 助词、语气词
    "的", "了", "吗", "呢", "吧", "啊", "呀", "哇", "哦", "嗯",
    "着", "过", "地", "得", "么", "嘛", "呵",

    # 连词
    "和", "与", "及", "或", "或者", "以及", "并", "并且", "而", "而且",
    "但", "但是", "可是", "然而", "不过", "却", "则", "就",
    "因为", "所以", "因此", "如果", "假如", "要是",

    # 介词
    "在", "于", "对", "对于", "关于", "由", "从", "自", "为", "向",
    "往", "朝", "沿", "随", "把", "被", "给", "将",

    # 副词（无实际意义的）
    "很", "最", "更", "太", "非常", "十分", "特别", "格外",
    "还", "也", "都", "又", "再", "才", "只",
    "已", "已经", "曾", "曾经", "正在", "刚刚",

    # 量词（通用的）
    "一个", "两个", "几个", "一些", "一点", "有些", "所有", "每个",

    # 动词（太泛化的）
    "是", "有", "在", "为", "以", "会", "能", "要", "可以",
    "做", "搞", "弄", "行", "来", "去", "到",

    # 时间词（泛化的）
    "时候", "时", "刚", "刚才", "现在", "当前", "以后", "之后",
    "以前", "之前", "过去", "将来", "未来",

    # 其他无意义词
    "等", "等等", "之类", "类", "型", "样", "种", "般",
    "如", "像", "比如", "譬如", "例如",
}


# ================= TF-IDF 关键词提取 =================

def tokenize(text):
    """使用 jieba 对中文进行分词，同时提取英数字符，过滤停用词"""
    chinese_tokens = [t for t in jieba.cut(text) if len(t) >= 2]
    english_tokens = re.findall(r'[A-Za-z0-9]+', text)
    tokens = chinese_tokens + english_tokens
    return [t for t in tokens if t not in STOP_WORDS and len(t) >= 2]


def extract_tfidf_keywords(target_text, all_texts, top_n=6):
    """基于 TF-IDF 提取目标片段的独特关键词"""
    total_docs = len(all_texts)
    target_tokens = tokenize(target_text)
    if not target_tokens:
        return []

    tf_counter = Counter(target_tokens)
    total_terms = len(target_tokens)

    doc_freq = {}
    for text in all_texts:
        for token in set(tokenize(text)):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    tfidf_scores = {}
    for term, count in tf_counter.items():
        tf = count / total_terms
        idf = math.log(total_docs / (doc_freq.get(term, 0) + 1)) + 1
        tfidf_scores[term] = tf * idf

    keywords = sorted(tfidf_scores, key=tfidf_scores.get, reverse=True)[:top_n]
    return keywords


def highlight_query_keywords(text, query):
    """检索结果高亮：提取问题中的核心实词，在片段中高亮"""
    chinese_tokens = [t for t in jieba.cut(query) if len(t) >= 2]
    english_tokens = re.findall(r'[A-Za-z0-9]+', query)
    query_tokens = [
        t for t in chinese_tokens + english_tokens
        if t not in STOP_WORDS and len(t) >= 2
    ]

    for w in query_tokens:
        text = re.sub(
            re.escape(w),
            f"<mark style='background:#ffe066;font-weight:bold'>{w}</mark>",
            text,
            flags=re.IGNORECASE
        )

    return text


# ================= 引用出处精确句子匹配高亮 =================

def sentence_overlap(sentence, answer):
    """计算原文中某句话与答案的字符级重叠度"""
    s_tokens = set(re.findall(r'[\u4e00-\u9fa5]|[A-Za-z0-9]+', sentence))
    a_tokens = set(re.findall(r'[\u4e00-\u9fa5]|[A-Za-z0-9]+', answer))
    if not s_tokens:
        return 0.0
    return len(s_tokens & a_tokens) / len(s_tokens)


def highlight_evidence_sentences(text, answer):
    """引用出处高亮：只高亮原文中真正被答案引用的部分"""
    answer_tokens = [
        t for t in jieba.cut(answer)
        if len(t.strip()) >= 2 and t not in STOP_WORDS
    ]

    if not answer_tokens:
        return text

    answer_keywords = set(answer_tokens)

    answer_patterns = set()
    answer_patterns.update(re.findall(
        r'\d+\.?\d*\s*(?:Pa|dB|分钟|小时|秒|天|mAh|Ah|mm|cm|m|米|厘米|毫米|'
        r'm²|平方米|ml|l|L|升|毫升|kg|g|克|千克|W|V|A|Hz|GHz|MHz|%)',
        answer, re.IGNORECASE
    ))
    answer_patterns.update(re.findall(r'\d{3,}', answer))
    answer_patterns.update(re.findall(r'[A-Za-z]+[0-9]+[A-Za-z0-9]*', answer))

    lines = text.split('\n')
    result = []
    highlighted_count = 0
    max_highlights = 3

    for line in lines:
        if not line.strip():
            result.append(line + '\n')
            continue

        line_tokens = [
            t for t in jieba.cut(line)
            if len(t.strip()) >= 2 and t not in STOP_WORDS
        ]
        line_keywords = set(line_tokens)

        keyword_overlap = len(answer_keywords & line_keywords)
        pattern_matches = sum(1 for p in answer_patterns if p in line)
        answer_text = ''.join(answer_keywords)
        line_text = ''.join(line_keywords)
        common_chars = set(answer_text) & set(line_text)
        char_overlap_ratio = len(common_chars) / max(len(set(answer_text)), 1)

        should_highlight = False
        if pattern_matches > 0 and keyword_overlap >= 1:
            should_highlight = True
        elif keyword_overlap >= max(2, len(answer_keywords) * 0.5):
            should_highlight = True
        elif char_overlap_ratio > 0.6 and keyword_overlap >= 1:
            should_highlight = True

        if len(line.strip()) < 10:
            should_highlight = False

        if should_highlight and highlighted_count < max_highlights:
            result.append(
                f"<mark style='background:#ffd54f;font-weight:bold;"
                f"border-radius:3px;padding:1px 2px;'>"
                f"{line}</mark>\n"
            )
            highlighted_count += 1
        else:
            result.append(line + '\n')

    return ''.join(result)


# ================= 引用脚标（可点击） =================

def style_refs(answer):
    def replace_ref(match):
        num = match.group(1)
        return f"<sup class='ref'><a href='#ref-{num}'>[{num}]</a></sup>"
    answer = re.sub(r"\[(\d+)\]", replace_ref, answer)
    return answer


# ================= 检索 =================

def get_sources(query):
    docs_scores = vectordb.similarity_search_with_score(query, k=top_k)
    results = []

    for doc, score in docs_scores:
        similarity = round(1 / (1 + score), 3)
        clean_content = re.sub(r'<[^>]+>', '', doc.page_content, flags=re.DOTALL)
        source_name = os.path.basename(doc.metadata.get("source", "未知"))
        file_type = get_ext(source_name)

        results.append({
            "content": clean_content,
            "source": source_name,
            "score": similarity,
            "file_type": file_type,
            "is_image_ocr": doc.metadata.get("type") == "image_ocr",
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results


# ================= LLM =================

llm = ChatTongyi(model="qwen-plus", temperature=0)

# ================= 聊天历史 =================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "嗨！我是文档问答小助手，随时为你解答文档相关问题～"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ================= 用户输入 =================

query = st.chat_input("请输入问题")

if query:

    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    if console_output:
        print("\n" + "=" * 80)
        print(f"📝 用户问题: {query}")
        print("=" * 80)

    with st.chat_message("assistant"):

        sources = get_sources(query)
        total_sources = len(sources)

        if console_output:
            print(f"\n🔍 检索到 Top-{top_k} 文档:")
            print("-" * 80)
            for i, src in enumerate(sources, start=1):
                print(f"\n【文档 {i}】")
                print(f"来源文件: {src['source']} | 类型: {src['file_type']}")
                ocr_tag = " [图片OCR]" if src['is_image_ocr'] else ""
                print(f"相似度得分: {src['score']:.3f} ({src['score'] * 100:.1f}%){ocr_tag}")
                print(f"内容预览 (前200字):\n{src['content'][:200]}...")
                print("-" * 80)

        context = ""
        for i, src in enumerate(sources, start=1):
            content = re.sub(r'[=＝]{2,}.*?[一二三四五六七八九十百]+.*?\n?', '', src['content'])
            content = re.sub(r'[一二三四五六七八九十百]+[、．\.]\s*', '', content)
            content = re.sub(r'第[一二三四五六七八九十百]+[节章条款]\s*', '', content)
            # 标注图片来源，帮助LLM理解
            source_tag = f"[来自图片OCR: {src['source']}]" if src['is_image_ocr'] else ""
            context += f"\n[{i}] {source_tag}{content}\n"

        rag_prompt = f"""
请根据以下文档内容回答问题。

规则：
1. 只根据文档内容作答，不要引入文档以外的信息。
2. 在引用文档内容时，必须在句末用阿拉伯数字方括号标注来源编号，例如[1]或[2]。
3. 引用编号只能使用1到{total_sources}之间的整数，不能使用中文数字或其他格式。
4. 如果文档中没有相关信息，请明确回答"文档中未提及相关信息"。

问题:
{query}

文档:
{context}
"""

        if console_output:
            print("\n📤 发送给 LLM 的 Prompt:")
            print("-" * 80)
            print(rag_prompt)
            print("-" * 80)

        answer = llm.invoke(rag_prompt).content

        if console_output:
            print("\n🤖 LLM 原始回答:")
            print("-" * 80)
            print(answer)
            print("-" * 80)

        # 清理中文数字编号
        chinese_num_map = {
            "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
            "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"
        }
        for cn, ar in chinese_num_map.items():
            answer = answer.replace(f"[{cn}]", f"[{ar}]")

        refs = re.findall(r"\[(\d+)\]", answer)
        used = sorted(set(n for n in map(int, refs) if 1 <= n <= total_sources))

        mapping = {old: i + 1 for i, old in enumerate(used)}
        for old, new in mapping.items():
            answer = answer.replace(f"[{old}]", f"[{new}]")

        answer = re.sub(r"引用来源[:：].*", "", answer)
        answer = re.sub(r"\n?\s*(\[\d+\]\s*)+\s*$", "", answer)

        if console_output:
            print("\n✅ 清理后的最终答案:")
            print("-" * 80)
            print(answer)
            print("-" * 80)
            print("\n📎 引用映射:")
            if mapping:
                for old, new in mapping.items():
                    src = sources[old - 1]
                    print(f"  [{new}] -> 原始文档 [{old}] ({src['source']})")
            else:
                print("  无引用")
            print("=" * 80 + "\n")

        styled_answer = style_refs(answer)
        st.markdown(styled_answer, unsafe_allow_html=True)

        # ===== 引用出处 =====

        st.markdown("## 📎 引用出处")

        if used:
            for old, new in mapping.items():
                src = sources[old - 1]
                display_content = src['content']
                icon = FILE_ICONS.get(src['file_type'], "📎")
                ocr_label = " · 图片OCR" if src['is_image_ocr'] else ""

                st.markdown(f"<div id='ref-{new}'></div>", unsafe_allow_html=True)

                with st.expander(f"[{new}] {icon} {src['source']}{ocr_label}"):
                    evidence = highlight_evidence_sentences(display_content, answer)
                    evidence = evidence.rstrip('\n')
                    st.markdown(
                        f"""<div style="background:#f8f9fa;padding:14px;border-radius:8px;line-height:1.8;font-size:15px;">{evidence}</div>""",
                        unsafe_allow_html=True
                    )

        else:
            st.info("暂无可溯源的引用片段，答案可能来自多个片段的综合推断。")

        # ===== 检索结果 =====

        if debug_mode:
            st.markdown("## 🔎 检索结果")

            chinese_tokens = [t for t in jieba.cut(query) if len(t) >= 2]
            english_tokens = re.findall(r'[A-Za-z0-9]+', query)
            query_tokens = [
                t for t in chinese_tokens + english_tokens
                if t not in STOP_WORDS and len(t) >= 2
            ]

            for i, src in enumerate(sources, start=1):
                score = round(src["score"] * 100, 1)
                icon = FILE_ICONS.get(src['file_type'], "📎")
                hit_words = [w for w in query_tokens if w in src["content"]]
                hit_label = "｜ 🔑 " + " · ".join(hit_words) if hit_words else ""
                ocr_label = " · 🖼️OCR" if src['is_image_ocr'] else ""

                with st.expander(
                    f"{i}. {icon} {src['source']} ｜ 相似度 {score}%{ocr_label}{hit_label}"
                ):
                    text = highlight_query_keywords(src["content"], query)
                    st.markdown(text, unsafe_allow_html=True)
                    st.progress(src["score"])