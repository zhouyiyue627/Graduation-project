# streamlit run LangChain-RAG-QA.py

import streamlit as st
import tempfile
import os
import re
import math
import jieba
from collections import Counter

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
    st.markdown("<div class='sidebar-title'>🔑 API配置</div>", unsafe_allow_html=True)

    api_key = st.text_input("API Key", type="password")

    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key

if not api_key:
    st.warning("请输入API Key")
    st.stop()

# ================= 上传 =================

st.sidebar.markdown("<div class='sidebar-title'>📂 上传TXT文件</div>", unsafe_allow_html=True)

uploaded_files = st.sidebar.file_uploader(
    "选择 TXT 文件",
    type=["txt"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("请上传TXT文件")
    st.stop()

# ================= 文档浏览 =================

st.sidebar.markdown("<div class='sidebar-title'>📑 文档浏览</div>", unsafe_allow_html=True)

for f in uploaded_files:
    with st.sidebar.expander(f.name):
        preview = f.getvalue().decode("utf-8")[:500]

        st.write(preview + "...")

# ================= 参数 =================

st.sidebar.markdown("<div class='sidebar-title'>⚙️ Top-K 检索数量</div>", unsafe_allow_html=True)

top_k = st.sidebar.slider(
    "检索文档数量",
    1, 10, 5
)

st.sidebar.markdown("<div class='sidebar-title'>🐞 RAG Debug</div>", unsafe_allow_html=True)

debug_mode = st.sidebar.checkbox("显示检索结果")

# ===== 新增：控制台输出开关 =====
st.sidebar.markdown("<div class='sidebar-title'>💻 PyCharm 控制台输出</div>", unsafe_allow_html=True)
console_output = st.sidebar.checkbox("启用控制台调试输出", value=False)


# ================= 构建向量库 =================

@st.cache_resource
def build_vector_db(uploaded_files, _file_hashes):
    """
    构建向量数据库
    _file_hashes: 用于缓存失效判断，当文件内容改变时重新构建
    """

    docs = []

    temp_dir = tempfile.TemporaryDirectory()

    for file in uploaded_files:
        path = os.path.join(temp_dir.name, file.name)

        with open(path, "wb") as f:
            f.write(file.getvalue())

        loader = TextLoader(path, encoding="utf-8")

        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = splitter.split_documents(docs)

    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2"
    )

    vectordb = Chroma.from_documents(
        splits,
        embeddings
    )

    return vectordb


# 生成文件哈希值用于缓存判断
file_hashes = tuple(hash(f.getvalue()) for f in uploaded_files)
vectordb = build_vector_db(uploaded_files, file_hashes)

# ================= 停用词 =================
# 停用词应该是：语气词、代词、助词、连词等无实际意义的词
# 保留技术关键词和用户提问核心词

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

    # 时间词（泛化的，保留"小时"、"分钟"等具体单位）
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
    """
    基于 TF-IDF 提取目标片段的独特关键词。

    TF（词频）：某词在当前片段中出现的频率
    IDF（逆文档频率）：log(总片段数 / 含该词的片段数) + 1
        - 在越少片段中出现的词，IDF 越高，说明越具有区分度
    TF-IDF = TF × IDF，值越高说明该词对当前片段越具代表性
    """
    total_docs = len(all_texts)

    # 计算当前片段的词频 TF
    target_tokens = tokenize(target_text)
    if not target_tokens:
        return []

    tf_counter = Counter(target_tokens)
    total_terms = len(target_tokens)

    # 计算每个词出现在多少个片段中（文档频率 DF）
    doc_freq = {}
    for text in all_texts:
        for token in set(tokenize(text)):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    # 计算 TF-IDF 分值
    tfidf_scores = {}
    for term, count in tf_counter.items():
        tf = count / total_terms
        idf = math.log(total_docs / (doc_freq.get(term, 0) + 1)) + 1
        tfidf_scores[term] = tf * idf

    # 按分值降序，取前 top_n 个关键词
    keywords = sorted(tfidf_scores, key=tfidf_scores.get, reverse=True)[:top_n]
    return keywords


def highlight_query_keywords(text, query):
    """
    检索结果高亮：提取问题中的核心实词，在片段中高亮与问题直接相关的词。

    核心思路：
    - 用户想知道的是"这个片段哪里和我的问题有关"
    - 因此高亮的应该是问题中的核心实词，而不是片段内部的统计关键词
    - 使用 jieba 精准分词，避免正则将整句中文匹配为一个词
    - 过滤停用词后，剩下的词才是用户真正关心的内容
    """
    # 使用 jieba 对问题进行精准分词，同时提取英数字符
    chinese_tokens = [t for t in jieba.cut(query) if len(t) >= 2]
    english_tokens = re.findall(r'[A-Za-z0-9]+', query)
    query_tokens = [
        t for t in chinese_tokens + english_tokens
        if t not in STOP_WORDS and len(t) >= 2
    ]

    for w in query_tokens:
        text = re.sub(
            re.escape(w),
            f"<mark style='background:#ffe066;"
            f"font-weight:bold'>{w}</mark>",
            text,
            flags=re.IGNORECASE
        )

    return text


# ================= 引用出处精确句子匹配高亮 =================

def sentence_overlap(sentence, answer):
    """
    计算原文中某句话与答案的字符级重叠度。
    重叠度 = 句子与答案共有的词 / 句子的词总数
    值越高说明该句子越可能是答案的引用依据。
    """
    s_tokens = set(re.findall(r'[\u4e00-\u9fa5]|[A-Za-z0-9]+', sentence))
    a_tokens = set(re.findall(r'[\u4e00-\u9fa5]|[A-Za-z0-9]+', answer))

    if not s_tokens:
        return 0.0

    return len(s_tokens & a_tokens) / len(s_tokens)


def highlight_evidence_sentences(text, answer):
    """
    引用出处高亮：只高亮原文中真正被答案引用的部分

    核心逻辑：
    1. 从答案中提取"有意义的连续文本片段"（去除停用词后的连续词组）
    2. 在原文中查找包含这些片段的句子
    3. 使用字符级重叠度判断，只高亮真正匹配的部分
    """

    # ===== 第一步：从答案中提取有意义的文本片段 =====
    # 使用 jieba 分词，提取非停用词
    answer_tokens = [
        t for t in jieba.cut(answer)
        if len(t.strip()) >= 2 and t not in STOP_WORDS
    ]

    if not answer_tokens:
        return text

    # 构建答案的关键词集合
    answer_keywords = set(answer_tokens)

    # 提取答案中的特征模式（数字+单位、产品型号等）
    answer_patterns = set()

    # 数字+单位组合
    answer_patterns.update(re.findall(
        r'\d+\.?\d*\s*(?:Pa|dB|分钟|小时|秒|天|mAh|Ah|mm|cm|m|米|厘米|毫米|'
        r'm²|平方米|ml|l|L|升|毫升|kg|g|克|千克|W|V|A|Hz|GHz|MHz|%)',
        answer,
        re.IGNORECASE
    ))

    # 3位以上的纯数字
    answer_patterns.update(re.findall(r'\d{3,}', answer))

    # 产品型号（包含字母和数字的组合）
    answer_patterns.update(re.findall(r'[A-Za-z]+[0-9]+[A-Za-z0-9]*', answer))

    # ===== 第二步：按行切分原文，逐行评估相关性 =====
    lines = text.split('\n')
    result = []
    highlighted_count = 0
    max_highlights = 3  # 最多高亮3处

    for line in lines:
        if not line.strip():
            result.append(line + '\n')
            continue

        # 对当前行进行分词
        line_tokens = [
            t for t in jieba.cut(line)
            if len(t.strip()) >= 2 and t not in STOP_WORDS
        ]
        line_keywords = set(line_tokens)

        # ===== 计算匹配度 =====

        # 1. 关键词重叠数量
        keyword_overlap = len(answer_keywords & line_keywords)

        # 2. 检查是否包含特征模式（数字+单位等）
        pattern_matches = sum(1 for p in answer_patterns if p in line)

        # 3. 计算字符级相似度（用于精确匹配）
        # 将答案和行都转换为纯关键字字符串进行比较
        answer_text = ''.join(answer_keywords)
        line_text = ''.join(line_keywords)

        # 计算简单的字符重叠率
        common_chars = set(answer_text) & set(line_text)
        char_overlap_ratio = len(common_chars) / max(len(set(answer_text)), 1)

        # ===== 判断是否应该高亮 =====
        # 条件：必须同时满足以下之一
        should_highlight = False

        # 条件1：包含特征模式（如"4500Pa"）且有关键词重叠
        if pattern_matches > 0 and keyword_overlap >= 1:
            should_highlight = True

        # 条件2：关键词重叠度很高（答案关键词的50%以上都在这行里）
        elif keyword_overlap >= max(2, len(answer_keywords) * 0.5):
            should_highlight = True

        # 条件3：字符重叠率很高
        elif char_overlap_ratio > 0.6 and keyword_overlap >= 1:
            should_highlight = True

        # 额外过滤：行太短的不高亮（可能是标题）
        if len(line.strip()) < 10:
            should_highlight = False

        # 限制高亮数量
        if should_highlight and highlighted_count < max_highlights:
            result.append(
                f"<mark style='background:#ffd54f;"
                f"font-weight:bold;"
                f"border-radius:3px;"
                f"padding:1px 2px;'>"
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
    docs_scores = vectordb.similarity_search_with_score(
        query,
        k=top_k
    )

    results = []

    for doc, score in docs_scores:
        similarity = 1 / (1 + score)
        similarity = round(similarity, 3)

        # 简单清理可能存在的 HTML 标签（防御性处理）
        clean_content = re.sub(r'<[^>]+>', '', doc.page_content, flags=re.DOTALL)

        results.append({
            "content": clean_content,
            "source": os.path.basename(
                doc.metadata.get("source", "未知")
            ),
            "score": similarity
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results


# ================= LLM =================

llm = ChatTongyi(
    model="qwen-plus",
    temperature=0
)

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

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    st.chat_message("user").write(query)

    # ===== 新增：控制台输出用户问题 =====
    if console_output:
        print("\n" + "=" * 80)
        print(f"📝 用户问题: {query}")
        print("=" * 80)

    with st.chat_message("assistant"):

        sources = get_sources(query)
        total_sources = len(sources)

        # ===== 新增：控制台输出检索到的 Top-K 文档 =====
        if console_output:
            print(f"\n🔍 检索到 Top-{top_k} 文档:")
            print("-" * 80)
            for i, src in enumerate(sources, start=1):
                print(f"\n【文档 {i}】")
                print(f"来源文件: {src['source']}")
                print(f"相似度得分: {src['score']:.3f} ({src['score'] * 100:.1f}%)")
                print(f"内容预览 (前200字):\n{src['content'][:200]}...")
                print("-" * 80)

        context = ""
        for i, src in enumerate(sources, start=1):
            # 直接删除片段内容中可能被 LLM 误读为引用编号的章节标记
            # 1. 删除 "====九、维护与保养====" 整行章节标题
            content = re.sub(r'[=＝]{2,}.*?[一二三四五六七八九十百]+.*?\n?', '', src['content'])
            # 2. 删除 "一、" "九、" 等中文序号前缀
            content = re.sub(r'[一二三四五六七八九十百]+[、．\.]\s*', '', content)
            # 3. 删除 "第一节" "第九章" 等带数字含义的章节标记
            content = re.sub(r'第[一二三四五六七八九十百]+[节章条款]\s*', '', content)
            context += f"\n[{i}] {content}\n"

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

        # ===== 新增：控制台输出发送给 LLM 的完整 Prompt =====
        if console_output:
            print("\n📤 发送给 LLM 的 Prompt:")
            print("-" * 80)
            print(rag_prompt)
            print("-" * 80)

        answer = llm.invoke(rag_prompt).content

        # ===== 新增：控制台输出 LLM 原始回答 =====
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

        # 提取引用编号，直接丢弃越界编号
        # 就近映射会导致无关片段被错误地作为引用出处展示
        refs = re.findall(r"\[(\d+)\]", answer)
        used = sorted(set(
            n for n in map(int, refs)
            if 1 <= n <= total_sources
        ))

        # 构建连续重映射
        mapping = {old: i + 1 for i, old in enumerate(used)}
        for old, new in mapping.items():
            answer = answer.replace(f"[{old}]", f"[{new}]")

        # 清理冗余内容
        answer = re.sub(r"引用来源[:：].*", "", answer)
        answer = re.sub(r"\n?\s*(\[\d+\]\s*)+\s*$", "", answer)

        # ===== 新增：控制台输出清理后的答案 =====
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
        # 精确句子匹配高亮：标注原文中与答案重叠度最高的句子

        st.markdown("## 📎 引用出处")

        if used:

            for old, new in mapping.items():
                # LLM 引用了 [n]，严格展示 sources[n-1]
                src = sources[old - 1]

                # content 已经在 get_sources() 中清理过 HTML 标签了
                display_content = src['content']

                st.markdown(
                    f"<div id='ref-{new}'></div>",
                    unsafe_allow_html=True
                )

                with st.expander(f"[{new}] 📄 {src['source']}"):
                    evidence = highlight_evidence_sentences(
                        display_content,
                        answer
                    )

                    # 去除末尾多余的换行符，避免 </div> 被当作文本显示
                    evidence = evidence.rstrip('\n')

                    st.markdown(
                        f"""<div style="background:#f8f9fa;padding:14px;border-radius:8px;line-height:1.8;font-size:15px;">{evidence}</div>""",
                        unsafe_allow_html=True
                    )

        else:
            # used 为空说明 LLM 未生成合法引用编号（如越界编号被丢弃）
            st.info("暂无可溯源的引用片段，答案可能来自多个片段的综合推断。")

        # ===== 检索结果 =====
        # 仅在勾选"显示检索结果"时展示

        if debug_mode:

            st.markdown("## 🔎 检索结果")

            # 提取问题核心实词（jieba精准分词），用于标题命中词展示与片段内高亮
            chinese_tokens = [t for t in jieba.cut(query) if len(t) >= 2]
            english_tokens = re.findall(r'[A-Za-z0-9]+', query)
            query_tokens = [
                t for t in chinese_tokens + english_tokens
                if t not in STOP_WORDS and len(t) >= 2
            ]

            for i, src in enumerate(sources, start=1):
                score = round(src["score"] * 100, 1)

                # 在标题中展示命中的关键词，让用户折叠状态下即可判断相关性
                hit_words = [w for w in query_tokens if w in src["content"]]
                hit_label = "｜ 🔑 " + " · ".join(hit_words) if hit_words else ""

                with st.expander(
                        f"{i}. {src['source']} ｜ 相似度 {score}%{hit_label}"
                ):
                    # 用问题核心实词高亮片段内容
                    text = highlight_query_keywords(src["content"], query)

                    st.markdown(text, unsafe_allow_html=True)

                    st.progress(src["score"])