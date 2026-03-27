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
from langchain_community.chat_models import ChatTongyi

# ================= 页面配置 =================

st.set_page_config(
    page_title="RAG 文档问答",
    layout="wide"
)

st.markdown("""
<style>
/* ===== 侧边栏字体层级 =====
   L1 功能块标题          : 15px 700  加粗主标题
   L2 控件标签 / 正文      : 13px 400  常规控件（checkbox/input/expander标题）
   L3 辅助提示 / 次要信息  : 12px 400 灰色  小提示/caption/已选tag/文本框输入
   ================================= */
section[data-testid="stSidebar"]{
    padding-top:10px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
/* L1：功能块标题（自定义div） */
.sidebar-title{
    font-size:15px !important;
    font-weight:700 !important;
    margin-top:16px;
    margin-bottom:6px;
    color: rgb(49, 51, 63) !important;
}
/* L2：所有控件label（checkbox/radio/input等）+ expander标题（文件预览文件名） */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] details summary p,
section[data-testid="stSidebar"] [data-testid="stCheckboxLabel"] {
    font-size:13px !important;
    font-weight:400 !important;
    color: rgb(49, 51, 63) !important;
    line-height:1.5 !important;
}
/* L3：辅助提示（caption/小文字）+ 所有次要信息 */
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
/* 文本框输入/占位符 */
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] textarea::placeholder,
/* 多选框已选tag */
section[data-testid="stSidebar"] [data-baseweb="tag"] span,
/* 数字输入框/按钮文字 */
section[data-testid="stSidebar"] input[type="number"],
section[data-testid="stSidebar"] button,
/* expander内部的次要文字（文件大小/提示） */
section[data-testid="stSidebar"] .stExpanderContent p,
section[data-testid="stSidebar"] .stExpanderContent .stCaption,
/* 滑块标签文字 */
section[data-testid="stSidebar"] [data-testid="stSliderLabel"] {
    font-size:12px !important;
    font-weight:400 !important;
    color:#888 !important;
    line-height:1.4 !important;
}
/* 修复滑块刻度文字大小 */
section[data-testid="stSidebar"] [data-testid="stSliderValue"] {
    font-size:12px !important;
}
/* 上传文件后显示的文件名/信息，与其他 L3 层级保持一致 */
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] p,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileData"] {
    font-size:12px !important;
    font-weight:400 !important;
    color:#888 !important;
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


# ================= 语义分块 =================

# 代码块围栏
_CODE_FENCE_RE = re.compile(r'^(`{3,}|~{3,})', re.MULTILINE)

# 句子边界
_SENTENCE_END_RE = re.compile(r'(?<=[。！？\.!\?])\s*')

# 每个 chunk 最大 token 数（写死，不暴露给用户）
_MAX_TOKENS = 500
_OVERLAP_TOKENS = 75
# 合并阈值：低于此 token 数的段落要向下合并
_MIN_TOKENS = 150


def _estimate_tokens(text: str) -> int:
    """粗估 token 数：中文 1字/token，英文约 4字符/token。"""
    chinese = len(re.findall(r'[\u4e00-\u9fa5]', text))
    others  = len(text) - chinese
    return chinese + max(0, others // 4)


def _split_by_sentences(text: str, max_tok: int, overlap_tok: int) -> list[str]:
    """
    对单个过长语义段按句子边界二次切分，并在子块之间保留 overlap。
    只有同一语义段内的子块才加 overlap，跨段不加。
    """
    sentences = [s for s in _SENTENCE_END_RE.split(text) if s.strip()]

    chunks, buf, buf_tok = [], [], 0
    for sent in sentences:
        sent_tok = _estimate_tokens(sent)
        if buf_tok + sent_tok > max_tok and buf:
            chunks.append(''.join(buf))
            # overlap：从末尾回溯
            overlap_buf, overlap_tok_count = [], 0
            for s in reversed(buf):
                t = _estimate_tokens(s)
                if overlap_tok_count + t > overlap_tok:
                    break
                overlap_buf.insert(0, s)
                overlap_tok_count += t
            buf, buf_tok = overlap_buf, overlap_tok_count
        buf.append(sent)
        buf_tok += _estimate_tokens(sent)

    if buf:
        chunks.append(''.join(buf))

    # 兜底：单句超长则按字符硬切
    result = []
    char_limit = max_tok * 3
    for c in chunks:
        if len(c) <= char_limit:
            result.append(c)
        else:
            for start in range(0, len(c), char_limit - overlap_tok * 2):
                result.append(c[start: start + char_limit])
    return result


def _heading_level(line: str) -> int:
    """
    判断标题行的层级（数字越小越高）。
    返回 0 表示不是标题。
    层级定义：
      1 = 最高：Markdown # / 第X章 / 汉字一级（一、二、三、）
      2 = 次级：Markdown ## / 第X节 / FAQ
      3 = 三级：Markdown ### 及更深
    """
    s = line.strip()
    # Markdown
    m = re.match(r'^(#{1,6})\s+\S', s)
    if m:
        depth = len(m.group(1))
        return min(depth, 3)
    # 第X章
    if re.match(r'^第[一二三四五六七八九十百千\d]+章\s*\S', s):
        return 1
    # 第X节/条/款/部分
    if re.match(r'^第[一二三四五六七八九十百千\d]+[节条款部分]\s*\S', s):
        return 2
    # 汉字一级：一、二、三、
    if re.match(r'^[一二三四五六七八九十]{1,3}[、]\s*\S', s):
        return 1
    # FAQ
    if re.match(r'^(?:Q|问|FAQ)\s*[：:]\s*\S', s):
        return 2
    return 0


def semantic_chunk_document(doc: Document) -> list[dict]:
    """
    对单篇文档做语义分块，返回 chunk 信息列表。
    每个元素：{ text, section_path, token_count, chunk_index }

    三阶段：
    1. 按章节标题切出语义段，维护多级标题栈，
       section_path 形如「第二章 总则 > 第三条 定义」
    2. 过短段落循环合并（< _MIN_TOKENS）
    3. 过长段落按句子二次切分（同段内加 overlap），子块路径加编号
    """
    text   = doc.page_content
    # 只取文件名，不要完整磁盘路径
    source = os.path.basename(doc.metadata.get("source", "未知文件"))

    # ── 第一阶段：识别结构边界，维护层级栈，切出语义段 ──────────
    lines = text.splitlines(keepends=True)
    raw_segments: list[tuple[str, str]] = []   # (section_path, content)

    # 层级栈：每个元素 (level, title_text)
    heading_stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_code        = False

    def current_path() -> str:
        """把栈里的标题拼成 文件名 > 章 > 节 > 条 的路径"""
        if not heading_stack:
            return source
        return source + " > " + " > ".join(t for _, t in heading_stack)

    for line in lines:
        # 代码块开关
        if _CODE_FENCE_RE.match(line):
            if not in_code:
                if buf:
                    raw_segments.append((current_path(), ''.join(buf)))
                    buf = []
                in_code = True
                buf = [line]
            else:
                buf.append(line)
                raw_segments.append((current_path() + " > [代码块]", ''.join(buf)))
                buf = []
                in_code = False
            continue

        if in_code:
            buf.append(line)
            continue

        level = _heading_level(line.strip())
        if level > 0:
            # 保存当前缓冲
            if buf:
                raw_segments.append((current_path(), ''.join(buf)))
                buf = []
            # 弹出层级 >= 当前的标题，维护栈
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            # 标题文字：去掉 # 前缀和首尾空格
            title_text = line.strip().lstrip('#').strip()
            heading_stack.append((level, title_text))
            buf = [line]
            continue

        buf.append(line)

    if buf:
        raw_segments.append((current_path(), ''.join(buf)))

    # ── 第二阶段：清洗 + 循环合并过短段落 ────────────────────────
    cleaned: list[tuple[str, str]] = []
    for sec, seg in raw_segments:
        seg = re.sub(r'(?m)^\s*\d+\s*$', '', seg)   # 纯页码行
        seg = re.sub(r'\n[=\-_]{3,}\n', '\n', seg)  # 分隔线
        seg = seg.strip()
        if seg:
            cleaned.append((sec, seg))

    def _same_parent(path_a: str, path_b: str) -> bool:
        """判断两个路径是否属于同一父章节（去掉最后一级后前缀相同）"""
        parts_a = path_a.split(" > ")
        parts_b = path_b.split(" > ")
        return parts_a[:-1] == parts_b[:-1]

    def _is_parent_of(path_a: str, path_b: str) -> bool:
        """判断 path_a 是否是 path_b 的直接父路径（章标题行合并到第一个子段）"""
        return path_b.startswith(path_a + " > ")

    def _merge_once(segs: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], bool]:
        merged, changed = [], False
        i = 0
        while i < len(segs):
            sec, seg = segs[i]
            tok = _estimate_tokens(seg)
            if tok < _MIN_TOKENS and i + 1 < len(segs):
                next_sec, next_seg = segs[i + 1]
                combined_tok = tok + _estimate_tokens(next_seg)
                # 合并条件：合并后不超长，且满足以下任一：
                # (a) 同一父章节内的短段
                # (b) 纯章节标题行（极短）向下合并到其第一个子段
                can_merge = combined_tok <= _MAX_TOKENS and (
                    _same_parent(sec, next_sec) or
                    (_is_parent_of(sec, next_sec) and tok < 20)
                )
                if can_merge:
                    merged.append((next_sec, seg + '\n' + next_seg))
                    i += 2
                    changed = True
                else:
                    merged.append((sec, seg))
                    i += 1
            else:
                merged.append((sec, seg))
                i += 1
        return merged, changed

    merged = cleaned
    for _ in range(20):
        merged, changed = _merge_once(merged)
        if not changed:
            break

    # ── 第三阶段：过长二次切分，子块路径加编号 ────────────────────
    chunks = []
    idx = 0
    for sec, seg in merged:
        tok = _estimate_tokens(seg)
        if tok <= _MAX_TOKENS:
            sub_list = [seg]
        else:
            sub_list = _split_by_sentences(seg, _MAX_TOKENS, _OVERLAP_TOKENS)

        total_subs = len(sub_list)
        for sub_i, sub in enumerate(sub_list):
            sub = sub.strip()
            if not sub:
                continue
            section_label = f"{sec} - {sub_i + 1}" if total_subs > 1 else sec
            chunks.append({
                "chunk_index":   idx,
                "section_path":  section_label,
                "token_count":   _estimate_tokens(sub),
                "text":          sub,
            })
            idx += 1

    # 兜底
    if not chunks:
        chunks.append({
            "chunk_index":  0,
            "section_path": source,
            "token_count":  _estimate_tokens(text),
            "text":         text.strip(),
        })

    return chunks


def chunks_to_documents(chunk_dicts: list[dict], base_meta: dict) -> list[Document]:
    """将 chunk dict 列表转为 LangChain Document，前缀拼接来源+章节。"""
    docs = []
    source = base_meta.get("source", "")
    for c in chunk_dicts:
        prefix  = f"【来源：{source}】【章节：{c['section_path']}】\n"
        content = prefix + c["text"]
        meta    = {
            **base_meta,
            "chunk_index":  c["chunk_index"],
            "section_path": c["section_path"],
            "token_count":  c["token_count"],
        }
        docs.append(Document(page_content=content, metadata=meta))
    return docs


# ================= 构建向量库 =================

@st.cache_resource
def build_vector_db(uploaded_files, _file_hashes, api_key):
    """
    构建向量数据库。
    _file_hashes 用于缓存失效（文件内容或 API Key 变化时重建）。
    分块参数固定，无需用户调整。
    """
    docs = []
    failed_files = []
    temp_dir = tempfile.TemporaryDirectory()

    progress = st.progress(0, text="正在处理文件...")
    total = len(uploaded_files)

    for i, file in enumerate(uploaded_files):
        ext  = get_ext(file.name)
        icon = FILE_ICONS.get(ext, "📎")
        progress.progress(
            (i + 1) / total,
            text=f"正在处理 {icon} {file.name} ({i+1}/{total})"
        )

        path = os.path.join(temp_dir.name, file.name)
        with open(path, "wb") as f:
            f.write(file.getvalue())

        loaded = load_single_file(path, file.name, api_key)

        for doc in loaded:
            if doc.metadata.get("error"):
                failed_files.append(file.name)
                break

        docs.extend(loaded)

    progress.empty()

    if failed_files:
        st.warning(f"以下文件加载时出现问题，请检查：{', '.join(failed_files)}")

    if not docs:
        st.error("所有文件均加载失败，无法构建向量库")
        st.stop()

    # 语义分块
    splits = []
    for doc in docs:
        chunk_dicts = semantic_chunk_document(doc)
        splits.extend(chunks_to_documents(chunk_dicts, doc.metadata))

    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    vectordb   = Chroma.from_documents(splits, embeddings)

    # 同时缓存每个文件的 chunk 预览数据，供侧边栏展示
    chunk_preview: dict[str, list[dict]] = {}
    for doc in docs:
        fname = os.path.basename(doc.metadata.get("source", ""))
        chunk_preview.setdefault(fname, []).extend(
            semantic_chunk_document(doc)
        )

    return vectordb, chunk_preview


# ================= 构建向量库（调用） =================

file_hashes = tuple(hash(f.getvalue()) for f in uploaded_files)
_result     = build_vector_db(uploaded_files, file_hashes, api_key)
vectordb, _chunk_preview = _result


# ================= 侧边栏：分块预览 =================

st.sidebar.markdown("<div class='sidebar-title'>📑 文档分块预览</div>", unsafe_allow_html=True)

for f in uploaded_files:
    ext  = get_ext(f.name)
    icon = FILE_ICONS.get(ext, "📎")
    chunks_for_file = _chunk_preview.get(f.name, [])
    total_chunks    = len(chunks_for_file)

    with st.sidebar.expander(f"{icon} {f.name}  · {total_chunks} 个 chunk"):
        if ext in IMAGE_TYPES:
            st.image(f.getvalue(), use_container_width=True)
            st.caption("🖼️ 图片已通过通义千问VL提取文字，分块结果如下")

        if not chunks_for_file:
            st.caption("暂无分块数据")
        else:
            for c in chunks_for_file:
                # section_path 已包含 文件名 > 章 > 节，直接显示
                display_path = c["section_path"]

                st.markdown(
                    f"<div style='"
                    f"background:#f0f4ff;border-left:3px solid #4a90e2;"
                    f"padding:6px 10px;border-radius:4px;margin-top:10px;"
                    f"font-size:12px;color:#444;line-height:1.6;'>"
                    f"<b>Chunk #{c['chunk_index'] + 1}</b> &nbsp;·&nbsp; "
                    f"{c['token_count']} tokens<br>"
                    f"<span style='color:#888;'>📂 {display_path}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                preview_text = c["text"][:300] + ("…" if len(c["text"]) > 300 else "")
                st.markdown(
                    f"<div style='"
                    f"background:#fafafa;border:1px solid #e0e0e0;"
                    f"padding:8px 10px;border-radius:4px;"
                    f"font-size:12px;color:#333;line-height:1.7;"
                    f"white-space:pre-wrap;word-break:break-all;'>"
                    f"{preview_text}"
                    f"</div>",
                    unsafe_allow_html=True
                )


# ================= 参数 =================

st.sidebar.markdown("<div class='sidebar-title'>⚙️ Top-K 检索数量</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size:12px;color:#888;margin:-4px 0 6px;'>召回最相关的 K 个文本块，建议值 3–6</div>", unsafe_allow_html=True)

top_k = st.sidebar.slider(
    "检索文档数量",
    1, 10, 5
)

st.sidebar.markdown("<div class='sidebar-title'>🐞 RAG Debug</div>", unsafe_allow_html=True)

debug_mode = st.sidebar.checkbox("显示检索结果")


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

    with st.chat_message("assistant"):

        sources = get_sources(query)
        total_sources = len(sources)

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

        answer = llm.invoke(rag_prompt).content

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