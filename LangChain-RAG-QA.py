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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ================= 页面配置 =================

st.set_page_config(
    page_title="RAG 文档问答",
    layout="wide"
)

st.markdown("""
<style>
/* =========================================================
   侧边栏字体层级（严格三级）
   L1  功能块标题  .sidebar-title : 14px 600  主色
   L2  控件标签   label / summary : 13px 400  主色   ← API Key、选择文件、检索数量、显示检索结果 等
   L3  辅助信息   caption / hint  : 12px 400  灰色   ← 提示文字、文件大小、分块信息
   按钮                           : 13px 500  主色   ← 清空对话等操作按钮与 L2 齐平
   ========================================================= */

section[data-testid="stSidebar"] {
    padding-top: 10px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* L1：功能块标题 */
.sidebar-title {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: rgb(49, 51, 63) !important;
    margin-top: 18px;
    margin-bottom: 6px;
    line-height: 1.4;
}

/* L2：所有控件标签 —— label、expander 摘要、checkbox 文字 */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] details > summary p,
section[data-testid="stSidebar"] [data-testid="stCheckboxLabel"] p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: 13px !important;
    font-weight: 400 !important;
    color: rgb(49, 51, 63) !important;
    line-height: 1.5 !important;
}

/* 操作按钮：与 L2 同级，略加粗以示可点击 */
section[data-testid="stSidebar"] button,
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
section[data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: rgb(49, 51, 63) !important;
}

section[data-testid="stSidebar"] button p,
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p {
    font-size: 13px !important;
    font-weight: 500 !important;
    line-height: normal !important;
}

/* L3：辅助提示、caption、文件上传说明、分块内容小字 */
section[data-testid="stSidebar"] .stCaption p,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] textarea::placeholder,
section[data-testid="stSidebar"] [data-baseweb="tag"] span,
section[data-testid="stSidebar"] input[type="number"],
section[data-testid="stSidebar"] .stExpanderContent p,
section[data-testid="stSidebar"] .stExpanderContent .stCaption,
section[data-testid="stSidebar"] [data-testid="stSliderLabel"] p,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] p,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileData"] {
    font-size: 12px !important;
    font-weight: 400 !important;
    color: #999 !important;
    line-height: 1.4 !important;
}

/* 滑块当前值数字 */
section[data-testid="stSidebar"] [data-testid="stSliderValue"] {
    font-size: 12px !important;
    color: rgb(49, 51, 63) !important;
}

/* 文件上传区"拖拽文件到此处" */
[data-testid="stFileUploaderDropzoneInstructions"] > div::before {
    content: "拖拽文件到此处";
    display: block;
    font-size: 13px;
    color: rgb(49, 51, 63);
    margin-bottom: 4px;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div::after {
    content: "每个文件限制 200MB";
    display: block;
    font-size: 12px;
    color: #999;
    margin-top: 2px;
}
.ewslnz93, .ewslnz94 { display: none !important; }

/* 文件上传按钮汉化 */
[data-testid="stFileUploaderDropzone"] button[data-testid="baseButton-secondary"],
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] {
    color: transparent !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stFileUploaderDropzone"] button[data-testid="baseButton-secondary"]::after,
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]::after {
    content: "浏览文件";
    color: rgb(49, 51, 63);
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    white-space: nowrap;
    pointer-events: none;
    font-size: 13px !important;
    font-weight: 500 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    line-height: normal !important;
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
</style>
""", unsafe_allow_html=True)

st.title("📚 文档问答系统")

# ================= 文件类型配置 =================

SUPPORTED_TYPES = ["txt", "pdf", "docx", "doc", "png", "jpg", "jpeg"]

FILE_ICONS = {
    "txt": "📄",
    "pdf": "📕",
    "docx": "📘",
    "doc": "📗",
    "png": "🖼️",
    "jpg": "🖼️",
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
                # PDF 是扫描件，尝试逐页 OCR
                if not api_key:
                    return [Document(
                        page_content=f"[{filename}：PDF内容为空（扫描件），请在侧边栏输入API Key以启用自动OCR]",
                        metadata={"source": filename}
                    )]
                try:
                    from pdf2image import convert_from_path
                    pages = convert_from_path(path, dpi=150)
                    ocr_docs = []
                    for i, page_img in enumerate(pages):
                        import io
                        buf = io.BytesIO()
                        page_img.save(buf, format="PNG")
                        img_bytes = buf.getvalue()
                        page_text = extract_text_from_image_qwen(img_bytes, f"{filename}_p{i+1}.png", api_key)
                        if page_text.strip():
                            ocr_docs.append(Document(
                                page_content=page_text,
                                metadata={"source": filename, "type": "image_ocr", "page": i + 1}
                            ))
                    if ocr_docs:
                        return ocr_docs
                    return [Document(
                        page_content=f"[{filename}：扫描件OCR未提取到文字内容，请检查图片质量]",
                        metadata={"source": filename}
                    )]
                except ImportError:
                    return [Document(
                        page_content=f"[{filename}：检测到扫描件PDF，自动OCR需要安装pdf2image库（pip install pdf2image）]",
                        metadata={"source": filename}
                    )]
                except Exception as e:
                    return [Document(
                        page_content=f"[{filename}：扫描件OCR处理失败：{str(e)}]",
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
    others = len(text) - chinese
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
    text = doc.page_content
    # 只取文件名，不要完整磁盘路径
    source = os.path.basename(doc.metadata.get("source", "未知文件"))

    # ── 第一阶段：识别结构边界，维护层级栈，切出语义段 ──────────
    lines = text.splitlines(keepends=True)
    raw_segments: list[tuple[str, str]] = []  # (section_path, content)

    # 层级栈：每个元素 (level, title_text)
    heading_stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_code = False

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
        seg = re.sub(r'(?m)^\s*\d+\s*$', '', seg)  # 纯页码行
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
                "chunk_index": idx,
                "section_path": section_label,
                "token_count": _estimate_tokens(sub),
                "text": sub,
            })
            idx += 1

    # 兜底
    if not chunks:
        chunks.append({
            "chunk_index": 0,
            "section_path": source,
            "token_count": _estimate_tokens(text),
            "text": text.strip(),
        })

    return chunks


def chunks_to_documents(chunk_dicts: list[dict], base_meta: dict) -> list[Document]:
    """将 chunk dict 列表转为 LangChain Document，前缀拼接来源+章节。"""
    docs = []
    source = base_meta.get("source", "")
    source = os.path.basename(source)
    for c in chunk_dicts:
        prefix = f"【来源：{source}】【章节：{c['section_path']}】\n"
        content = prefix + c["text"]
        meta = {
            **base_meta,
            "chunk_index": c["chunk_index"],
            "section_path": c["section_path"],
            "token_count": c["token_count"],
        }
        docs.append(Document(page_content=content, metadata=meta))
    return docs


# ================= 停用词（提前定义，BM25/tokenize 需要用）=================

STOP_WORDS = {
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
    "自己", "大家", "咱们", "某些", "另外",
    "这", "那", "这个", "那个", "这些", "那些", "这样", "那样",
    "此", "彼", "这里", "那里", "这儿", "那儿",
    "什么", "谁", "哪", "哪个", "哪些", "哪里", "哪儿",
    "的", "了", "吗", "呢", "吧", "啊", "呀", "哇", "哦", "嗯",
    "着", "过", "地", "得", "么", "嘛", "呵",
    "和", "与", "及", "或", "或者", "以及", "并", "并且", "而", "而且",
    "但", "但是", "可是", "然而", "不过", "却", "则", "就",
    "因为", "所以", "因此", "如果", "假如", "要是",
    "在", "于", "对", "对于", "关于", "由", "从", "自", "为", "向",
    "往", "朝", "沿", "随", "把", "被", "给", "将",
    "很", "最", "更", "太", "非常", "十分", "特别", "格外",
    "还", "也", "都", "又", "再", "才", "只",
    "已", "已经", "曾", "曾经", "正在", "刚刚",
    "一个", "两个", "几个", "一些", "一点", "有些", "所有", "每个",
    "是", "有", "在", "为", "以", "会", "能", "要", "可以",
    "做", "搞", "弄", "行", "来", "去", "到",
    "时候", "时", "刚", "刚才", "现在", "当前", "以后", "之后",
    "以前", "之前", "过去", "将来", "未来",
    "等", "等等", "之类", "类", "型", "样", "种", "般",
    "如", "像", "比如", "譬如", "例如",
}


def tokenize(text: str) -> list[str]:
    """jieba 分词 + 英文提取，过滤停用词，供 BM25 和高亮复用。"""
    chinese_tokens = [t for t in jieba.cut(text) if len(t) >= 2]
    english_tokens = re.findall(r'[A-Za-z0-9]+', text)
    tokens = chinese_tokens + english_tokens
    return [t for t in tokens if t not in STOP_WORDS and len(t) >= 2]


# ================= BM25 内联实现（无需额外依赖）=================

class BM25:
    """标准 BM25Okapi，中英文混合，内联实现避免外部依赖。"""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.corpus_size, 1)

        self.doc_freqs: list[dict] = []
        self.doc_lens: list[int] = []
        df: dict[str, int] = {}

        for doc in corpus:
            freq: dict[str, int] = {}
            for token in doc:
                freq[token] = freq.get(token, 0) + 1
            self.doc_freqs.append(freq)
            self.doc_lens.append(len(doc))
            for token in freq:
                df[token] = df.get(token, 0) + 1

        self.idf: dict[str, float] = {
            t: math.log(1 + (self.corpus_size - f + 0.5) / (f + 0.5))
            for t, f in df.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for i, freq in enumerate(self.doc_freqs):
                tf = freq.get(token, 0)
                if tf == 0:
                    continue
                dl = self.doc_lens[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf_val * numerator / denominator
        return scores


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
        ext = get_ext(file.name)
        icon = FILE_ICONS.get(ext, "📎")
        progress.progress(
            (i + 1) / total,
            text=f"正在处理 {icon} {file.name} ({i + 1}/{total})"
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
    vectordb = Chroma.from_documents(splits, embeddings)

    # 构建 BM25 索引（与向量库并行，用于混合检索）
    bm25_corpus = [tokenize(doc.page_content) for doc in splits]
    bm25_index = BM25(bm25_corpus)

    # 同时缓存每个文件的 chunk 预览数据，供侧边栏展示
    chunk_preview: dict[str, list[dict]] = {}
    for doc in docs:
        fname = os.path.basename(doc.metadata.get("source", ""))
        chunk_preview.setdefault(fname, []).extend(
            semantic_chunk_document(doc)
        )

    return vectordb, chunk_preview, bm25_index, splits


# ================= 构建向量库（调用） =================

file_hashes = tuple(hash(f.getvalue()) for f in uploaded_files)
_result = build_vector_db(uploaded_files, file_hashes, api_key)
vectordb, _chunk_preview, bm25_index, all_splits = _result

# ================= 侧边栏：分块预览 =================

st.sidebar.markdown("<div class='sidebar-title'>📑 文档分块预览</div>", unsafe_allow_html=True)

for f in uploaded_files:
    ext = get_ext(f.name)
    icon = FILE_ICONS.get(ext, "📎")
    chunks_for_file = _chunk_preview.get(f.name, [])
    total_chunks = len(chunks_for_file)

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
st.sidebar.markdown(
    "<div style='font-size:12px;color:#888;margin:-4px 0 6px;'>召回最相关的 K 个文本块，建议值 3–6</div>",
    unsafe_allow_html=True)

top_k = st.sidebar.slider(
    "检索文档数量",
    1, 10, 5
)

st.sidebar.markdown("<div class='sidebar-title'>🐞 RAG Debug</div>", unsafe_allow_html=True)

debug_mode = st.sidebar.checkbox("显示检索结果")

# ── 实验对比开关（论文测试用，测完可删除） ──
st.sidebar.markdown("<div class='sidebar-title'>🧪 实验对比（论文测试）</div>", unsafe_allow_html=True)
st.sidebar.caption("仅用于论文实验数据采集，正常使用请保持默认关闭")
use_vector_only = st.sidebar.checkbox("仅向量检索（关闭BM25+RRF）", value=False)
use_no_rewrite = st.sidebar.checkbox("关闭查询改写（使用原始问题）", value=False)

# 清空对话按钮
st.sidebar.markdown("<div class='sidebar-title'>💬 对话管理</div>", unsafe_allow_html=True)
if st.sidebar.button("🗑️ 清空对话", use_container_width=True):
    st.session_state.messages = [
        {"role": "assistant", "content": "对话已清空，随时可以重新提问～"}
    ]
    st.rerun()


# ================= TF-IDF 关键词提取 =================

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
    """
    引用出处高亮。

    核心改进：
    1. 切分策略：先按 \\n 切行，行内再按句子边界（。！？；·•）切分
       ·/• 是产品说明书常见条目分隔符，必须作为切分边界
    2. 匹配策略：用字符级汉字集合重叠度，比词语匹配更鲁棒
       不依赖 jieba 分词结果的准确性，直接算字符交集比例
    3. 阈值：单元字符集与答案字符集重叠 >= 35% 即高亮
    """
    # 从答案中提取所有有效汉字和英数字符（去停用词）
    answer_chars = set(re.findall(r'[\u4e00-\u9fa5]', answer))
    answer_words = set(re.findall(r'[A-Za-z0-9]{2,}', answer))

    # 数字+单位模式（精确匹配优先）
    answer_patterns = set()
    answer_patterns.update(re.findall(
        r'\d+\.?\d*\s*(?:Pa|dB|分钟|小时|秒|天|mAh|Ah|mm|cm|m|米|厘米|毫米|'
        r'm²|平方米|ml|l|L|升|毫升|kg|g|克|千克|W|V|A|Hz|GHz|MHz|%)',
        answer, re.IGNORECASE
    ))
    answer_patterns.update(re.findall(r'\d{3,}', answer))
    answer_patterns.update(re.findall(r'[A-Za-z]+[0-9]+[A-Za-z0-9]*', answer))

    if not answer_chars and not answer_words:
        return text

    # ── 切分：先按换行，再按 。！？；·• 切分条目 ──────────────────
    # 注意：·• 是条目符号，切分后本身留在对应条目里
    _unit_re = re.compile(r'(?<=[。！？；\.!\?])\s*|(?=[\·•])')

    def split_to_units(raw: str) -> list[str]:
        units = []
        for line in raw.split('\n'):
            stripped = line.strip()
            if not stripped:
                units.append('')
                continue
            # 按句子和条目符号切分
            parts = [p.strip() for p in _unit_re.split(stripped) if p.strip()]
            units.extend(parts if parts else [stripped])
        return units

    units = split_to_units(text)

    result = []
    highlighted_count = 0
    max_highlights = 6

    for unit in units:
        if not unit:
            result.append('\n')
            continue

        unit_chars = set(re.findall(r'[\u4e00-\u9fa5]', unit))
        unit_words = set(re.findall(r'[A-Za-z0-9]{2,}', unit))
        pattern_hit = any(p in unit for p in answer_patterns)

        # 汉字集合重叠率（核心指标）
        if answer_chars:
            char_ratio = len(answer_chars & unit_chars) / len(answer_chars)
        else:
            char_ratio = 0.0

        # 英数词重叠
        word_hit = bool(answer_words & unit_words)

        should_highlight = False
        if pattern_hit:
            should_highlight = True
        elif char_ratio >= 0.35:  # 答案里35%以上的汉字出现在该单元
            should_highlight = True
        elif char_ratio >= 0.20 and word_hit:
            should_highlight = True

        if len(unit) < 6:  # 过短片段（条目符号本身等）不高亮
            should_highlight = False

        if should_highlight and highlighted_count < max_highlights:
            result.append(
                f"<mark style='background:#ffd54f;font-weight:bold;"
                f"border-radius:3px;padding:1px 2px;'>"
                f"{unit}</mark>"
            )
            highlighted_count += 1
        else:
            result.append(unit)

    # 用换行重新拼接（单元已不含换行）
    return '\n'.join(result)


# ================= 引用脚标（可点击） =================

def style_refs(answer):
    def replace_ref(match):
        num = match.group(1)
        return f"<sup class='ref'><a href='#ref-{num}'>[{num}]</a></sup>"

    answer = re.sub(r"\[(\d+)\]", replace_ref, answer)
    return answer


# ================= 混合检索：BM25 + 向量 + RRF =================

def rrf_fusion(vec_ranked: list, bm25_ranked: list, k: int = 60) -> list:
    """
    Reciprocal Rank Fusion：合并两路排名列表。
    输入：各路按分数排好序的 (index, score) 列表。
    输出：按 RRF 分数降序排列的 index 列表。
    """
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(vec_ranked):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(bm25_ranked):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)


def get_sources(query: str) -> list[dict]:
    # ── 向量检索路 ──────────────────────────────────────────────
    vec_results = vectordb.similarity_search_with_score(query, k=top_k * 2)
    vec_ranked = []
    for doc, dist in vec_results:
        for i, split in enumerate(all_splits):
            if split.page_content == doc.page_content:
                vec_ranked.append((i, dist))
                break

    # ── 仅向量检索模式（对比实验用） ────────────────────────────
    if use_vector_only:
        fused_indices = [idx for idx, _ in sorted(vec_ranked, key=lambda x: x[1])][:top_k]
        bm25_scores = [0.0] * len(all_splits)
    else:
        # ── BM25 检索路 ──────────────────────────────────────────
        query_tokens = tokenize(query)
        bm25_scores = bm25_index.get_scores(query_tokens)
        bm25_top_idx = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:top_k * 2]
        bm25_ranked = [(i, bm25_scores[i]) for i in bm25_top_idx]
        # ── RRF 融合，取 Top-K ────────────────────────────────────
        fused_indices = rrf_fusion(vec_ranked, bm25_ranked)[:top_k]

    # ── 组装结果 ─────────────────────────────────────────────────
    results = []
    for idx in fused_indices:
        if idx >= len(all_splits):
            continue
        doc = all_splits[idx]
        # 优先用向量相似度作为展示分，不在向量结果里则用 BM25 归一化分
        vec_score = next((round(1 / (1 + d), 3) for i, d in vec_ranked if i == idx), None)
        bm25_raw = bm25_scores[idx] if idx < len(bm25_scores) else 0.0
        display_score = vec_score if vec_score is not None else round(min(bm25_raw / 10, 1.0), 3)

        clean_content = re.sub(r'<[^>]+>', '', doc.page_content, flags=re.DOTALL)
        source_name = os.path.basename(doc.metadata.get("source", "未知"))
        file_type = get_ext(source_name)

        results.append({
            "content": clean_content,
            "source": source_name,
            "score": display_score,
            "file_type": file_type,
            "is_image_ocr": doc.metadata.get("type") == "image_ocr",
        })

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

        # ── 自我介绍拦截（意图判断，避免误触发） ──────────────────
        # 正向触发：问题主语明确指向系统/助手本身
        intro_phrases = [
            "你是谁", "你是什么", "你叫什么", "你能做什么", "你可以做什么",
            "你有什么功能", "你的功能是", "这个系统是什么", "这个系统能做什么",
            "这个助手", "怎么使用这个", "如何使用你", "你支持什么格式",
            "你怎么用", "使用说明", "能干啥", "你能干什么",
            "介绍一下你", "介绍你自己", "介绍一下这个系统", "帮我介绍你自己"
        ]
        # 负向条件：包含文档内容相关词则说明用户在问文档，不应拦截
        doc_related_words = [
            "文档", "说明书", "手册", "产品", "功能介绍", "操作", "设置",
            "参数", "规格", "电池", "吸力", "过滤", "充电", "清洁", "故障",
            "维修", "保修", "安装", "连接", "APP", "模式", "传感器", "地毯",
            "集尘", "过滤网", "边刷", "主刷", "导航", "建图", "续航"
        ]
        is_intro_query = (
            any(phrase in query for phrase in intro_phrases)
            and not any(dw in query for dw in doc_related_words)
        )
        if is_intro_query:
            intro = "你好！我是文档问答小助手 📚\n\n快速开始：在左侧输入API Key，上传文档，然后直接提问即可。\n\n侧边栏功能：文档分块预览、Top-K检索数量调节、RAG Debug模式、清空对话。\n\n我支持精准定位文档内容并标注引用来源，支持多文档检索和多轮对话。"
            st.markdown("""
<div style='line-height:2;'>
<p>我可以帮你快速从上传的文档中找到答案，支持 TXT、PDF、Word、图片等格式。</p>
<p><b>🚀 快速开始：</b></p>
<ol style='margin-top:-8px;'>
  <li>在左侧侧边栏输入 API Key</li>
  <li>上传你的文档（支持 TXT、PDF、Word、图片）</li>
  <li>直接在下方输入框提问即可～</li>
</ol>
<p><b>📋 侧边栏功能说明：</b></p>
<ul style='margin-top:-8px;'>
  <li><b>文档分块预览</b>：查看文档被切分成了哪些片段</li>
  <li><b>Top-K 检索数量</b>：控制每次检索召回的文档片段数，建议 3–6</li>
  <li><b>RAG Debug</b>：开启后可查看检索到的原始片段及相似度评分</li>
  <li><b>清空对话</b>：清除当前对话历史，重新开始</li>
</ul>
<p><b>✨ 我的能力包括：</b></p>
<ul style='margin-top:-8px;'>
  <li>精准定位文档中的相关内容，并标注引用来源</li>
  <li>支持多文档同时检索</li>
  <li>支持追问和多轮对话</li>
</ul>
<p>有什么问题尽管问我吧！</p>
</div>
""", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": intro})
            st.stop()

        # Query Rewriting：把口语化问题改写为更适合检索的专业表述
        if use_no_rewrite:
            search_query = query  # 直接使用原始问题，跳过改写
        else:
            rewrite_prompt = (
                    "将以下用户问题改写为更适合文档检索的专业表述，"
                    "只输出改写后的问题，不要任何解释：\n" + query
            )
            search_query = llm.invoke(rewrite_prompt).content.strip()

        # 用改写后的 query 做混合检索
        sources = get_sources(search_query)
        total_sources = len(sources)

        # 置信度警告：最高分低于阈值时提示用户
        max_score = sources[0]["score"] if sources else 0.0
        CONFIDENCE_THRESHOLD = 0.30
        if max_score < CONFIDENCE_THRESHOLD:
            st.warning(
                f"⚠️ 文档中未找到高度相关内容（最高相似度 {max_score:.0%}），"
                "以下回答可能不够准确，请谨慎参考。"
            )

        cleaned_sources = []
        context = ""
        for i, src in enumerate(sources, start=1):
            content = re.sub(r'[=＝]{2,}.*?[一二三四五六七八九十百]+.*?\n?', '', src['content'])
            content = re.sub(r'[一二三四五六七八九十百]+[、．\.]\s*', '', content)
            content = re.sub(r'第[一二三四五六七八九十百]+[节章条款]\s*', '', content)
            content = re.sub(r'^【来源：[^】]*】【章节：[^】]*】\n?', '', content)
            cleaned_sources.append(content)
            source_tag = f"[来自图片OCR: {src['source']}]" if src['is_image_ocr'] else ""
            context += f"\n[{i}] {source_tag}{content}\n"

        # CoT 结构化推理 Prompt
        rag_prompt = f"""
请根据以下文档内容回答问题，并严格按指定格式输出。

回答规则：
1. 只根据文档内容作答，不要引入文档以外的信息。
2. 在引用文档内容时，必须紧跟在所引用的那句话末尾标注来源编号，例如[1]或[2]。
   每句话引用哪个片段，编号就紧跟在那句话后面，不得把所有编号堆在段落末尾。
   例如正确格式："定时清洁可设置7条[1]。静音模式噪音约52dB[2]。"
   错误格式："定时清洁可设置7条，静音模式噪音约52dB[1][2]。"
3. 引用编号只能使用1到{total_sources}之间的整数，不能使用中文数字或其他格式。
4. 如果文档中没有相关信息，请直接回复："抱歉，这个问题在已上传的文档中暂时没有找到相关信息，建议您查阅其他资料或上传更多相关文件～"，不要输出任何其他内容。
5. 直接给出清晰、完整的答案，不要在回答中暴露推理过程或分析步骤。
6. 如果问题涉及多个方面（如既问操作方法又问参数设置），每个方面必须分别找到
   对应文档片段并单独引用，不得遗漏任何一个相关来源。
7. 如果用户询问"你是什么"、"你能做什么"、"介绍一下你自己"、"这个系统是什么"
   等类似问题，请直接回复以下内容，不要输出ANSWER/EVIDENCE标签：
   "你好！我是文档问答小助手 📚

   我可以帮你快速从上传的文档中找到答案，支持 TXT、PDF、Word、图片等格式。
   你只需要上传文档，然后直接提问就好啦～

   我的能力包括：
   · 精准定位文档中的相关内容
   · 支持多文档同时检索
   · 引用时标注来源，方便核查
   · 支持追问和多轮对话

   有什么问题尽管问我吧！"

内部推理提示（不要在回答中体现）：
- 先找出与问题最直接相关的文档片段
- 如涉及多跳（A依赖B依赖C），在脑中逐步推导后再输出结论
- 如需比较或计数，先枚举再得出结论

输出格式（只输出以下两个标签，不要有任何其他文字）：

<ANSWER>
在这里写答案，含引用编号
</ANSWER>
<EVIDENCE>
[
  {{"src": 1, "text": "从文档[1]逐字摘录的原文片段1"}},
  {{"src": 1, "text": "从文档[1]逐字摘录的原文片段2"}},
  {{"src": 2, "text": "从文档[2]逐字摘录的原文片段"}}
]
</EVIDENCE>

EVIDENCE填写规则（非常重要）：
- text必须是从文档原文中直接复制粘贴的连续字符串，必须能在文档中一字不差地找到
- 禁止合并两处原文、禁止省略中间内容、禁止改写任何字符（包括标点）
- 如果原文有换行，也必须原样保留，不要替换为空格
- 单条 text 不超过25个汉字，超过必须拆分成多条
- 每一个独立的功能名称、数字结论、操作步骤，都必须单独列一条记录
- 答案中每引用一处原文，EVIDENCE里就对应一条记录，有几处引用就写几条
- 如果答案引用了某段中的多个子项（如列举了A、B、C三点），则A、B、C每条子项各自单独列一条记录
- 同一个src可以有多条记录

问题：
{query}

文档：
{context}
"""

        # 多轮追问：构建含历史的消息列表（保留最近 3 轮 = 6 条）
        history_msgs = []
        recent = [m for m in st.session_state.messages if m["role"] != "system"][:-1][-6:]
        for m in recent:
            if m["role"] == "user":
                history_msgs.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                history_msgs.append(AIMessage(content=m["content"]))

        system_msg = SystemMessage(content="你是一个严格基于文档回答问题的助手，不引用文档以外的知识。")
        final_msgs = [system_msg] + history_msgs + [HumanMessage(content=rag_prompt)]

        raw_output = llm.invoke(final_msgs).content

        # ── 解析 <ANSWER> ──────────────────────────────────────────
        answer_match = re.search(r'<ANSWER>(.*?)</ANSWER>', raw_output, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else raw_output.strip()

        # ── 解析 <EVIDENCE>：{原始src编号 -> [原文片段列表]} ────────
        import json

        evidence_by_src: dict[int, list[str]] = {}
        evidence_match = re.search(r'<EVIDENCE>(.*?)</EVIDENCE>', raw_output, re.DOTALL)
        if evidence_match:
            try:
                raw_json = evidence_match.group(1).strip()
                # 去掉 markdown 代码块围栏
                raw_json = re.sub(r'```[a-z]*\n?', '', raw_json).strip('`').strip()
                # 截取到 JSON 数组结束位置
                bracket_end = raw_json.rfind(']')
                if bracket_end != -1:
                    raw_json = raw_json[:bracket_end + 1]
                evidence_list = json.loads(raw_json)
                for item in evidence_list:
                    s = int(item.get("src", 0))
                    t = item.get("text", "").strip()
                    if s > 0 and t:
                        evidence_by_src.setdefault(s, []).append(t)
            except Exception:
                pass  # 解析失败时退化为无高亮

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
            # evidence_by_src 的 key 也要同步重映射
            if old in evidence_by_src:
                evidence_by_src[new] = evidence_by_src.pop(old)

        answer = re.sub(r"引用来源[:：].*", "", answer)
        answer = re.sub(r"\n?\s*(\[\d+\]\s*)+\s*$", "", answer)

        styled_answer = style_refs(answer)
        st.markdown(styled_answer, unsafe_allow_html=True)

        # 保存到对话历史（供下一轮多轮追问使用）
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # ===== 引用出处 =====

        no_info = "抱歉" in answer or not used

        if not no_info:
            st.markdown("## 📎 引用出处")

            if used:
                for old, new in mapping.items():
                    src = sources[old - 1]
                    raw_content = src['content']
                    display_content = re.sub(r'^【来源：[^】]*】【章节：[^】]*】\n?', '', raw_content)
                    icon = FILE_ICONS.get(src['file_type'], "📎")
                    ocr_label = " · 图片OCR" if src['is_image_ocr'] else ""

                    st.markdown(f"<div id='ref-{new}'></div>", unsafe_allow_html=True)

                    with st.expander(f"[{new}] {icon} {src['source']}{ocr_label}"):
                        fragments = evidence_by_src.get(new, [])
                        highlighted = cleaned_sources[old - 1]

                        if fragments:
                            for frag in sorted(fragments, key=len, reverse=True):
                                frag = frag.strip()
                                if len(frag) < 4:
                                    continue
                                if re.search(re.escape(frag), highlighted):
                                    highlighted = re.sub(
                                        re.escape(frag),
                                        f"<mark style='background:#ffd54f;font-weight:bold;"
                                        f"border-radius:3px;padding:1px 3px;'>{frag}</mark>",
                                        highlighted,
                                        count=1
                                    )
                                else:
                                    sub_phrases = re.split(r'[，。；：、！？,.;:!?\s]+', frag)
                                    for phrase in sub_phrases:
                                        phrase = phrase.strip()
                                        if len(phrase) < 4:
                                            continue
                                        if re.search(re.escape(phrase), highlighted):
                                            highlighted = re.sub(
                                                re.escape(phrase),
                                                f"<mark style='background:#ffd54f;font-weight:bold;"
                                                f"border-radius:3px;padding:1px 3px;'>{phrase}</mark>",
                                                highlighted,
                                                count=1
                                            )

                        st.markdown(
                            f"<div style='background:#f8f9fa;padding:14px;border-radius:8px;"
                            f"line-height:1.8;font-size:15px;white-space:pre-wrap;'>"
                            f"{highlighted}</div>",
                            unsafe_allow_html=True
                        )

        # ===== 检索结果 =====

        if debug_mode:
            st.markdown("## 🔎 检索结果（混合检索）")

            # 显示改写前后的对比
            if search_query != query:
                st.caption(f"🔁 问题改写：{query}  ➡️  {search_query}")

            # ── 检索评估指标（Hit Rate / MRR / Top-1相似度） ──────────
            # 从 EVIDENCE 中提取系统实际引用的片段索引（0-based）
            cited_indices = set()
            for ev_list in evidence_by_src.values():
                for ev_text in ev_list:
                    for j, src in enumerate(sources):
                        if ev_text.strip() and ev_text.strip() in src["content"]:
                            cited_indices.add(j)
                            break

            top1_score = sources[0]["score"] if sources else 0.0
            # Hit Rate：Top-K 中是否有被引用的片段
            hit = len(cited_indices) > 0
            # MRR：被引用片段中排名最高的倒数排名
            if cited_indices:
                best_rank = min(cited_indices) + 1   # 1-based
                mrr = round(1.0 / best_rank, 3)
            else:
                mrr = 0.0

            with st.expander("📊 检索评估指标（基于本次问答自动计算）", expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.metric("Top-1 相似度", f"{top1_score:.1%}")
                c2.metric("命中率 Hit Rate", "✅ 命中" if hit else "❌ 未命中",
                          help="答案引用的片段是否出现在 Top-K 检索结果中")
                c3.metric("MRR", f"{mrr:.3f}",
                          help="答案引用片段排名的倒数均值（越高越好，最大为1）")
                st.caption(
                    "注：以上指标基于本次问答的检索结果与答案引用自动计算，"
                    "不依赖人工标注。Hit Rate 和 MRR 以 EVIDENCE 片段的实际召回位置为依据。"
                )

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