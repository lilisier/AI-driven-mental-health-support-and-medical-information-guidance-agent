# 1. 导入必要的模块
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA  # 尽管会自定义逻辑，但保留用于潜在的直接检索能力
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import os
import openai  # 保留用于环境变量配置
import httpx  # 保留用于潜在的调试或高级配置

# --- API 配置 ---
base_url = "https://api.chatanywhere.tech/v1/"
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("错误：未找到 OPENAI_API_KEY 环境变量。请确保已正确配置。")
    exit()

# --- ChromaDB 持久化路径 ---
CHROMA_DB_PATH = "./vectordb"  # 定义向量数据库的保存路径

print("--- 步骤1: 加载文档 (模拟医疗知识库) ---")
# 2. 加载你的文档
# 为了演示医疗信息引导，这里假设 product_info.txt 包含了一些心理健康或医疗的科普信息
# 在实际项目中，你会构建一个更全面的医疗知识库
loader = TextLoader("青少年心理健康教育知识科普.txt", encoding="utf-8")
documents = loader.load()
print(f"成功加载 {len(documents)} 份文档，这些文档将作为我们的医疗知识库。")

print("\n--- 步骤2: 分割文档 ---")
# 3. 分割文档
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
print(f"文档被分割成 {len(texts)} 个文本块。")

print("\n--- 步骤3: 创建嵌入模型和向量数据库 ---")
# 4. 创建嵌入模型 (Embeddings)
embeddings = OpenAIEmbeddings(
    openai_api_base=base_url,
    model="text-embedding-ada-002",  # 确保此模型被你的第三方服务支持
    openai_api_key=api_key  # 传递API Key
)
print("嵌入模型初始化完成。")

# 5. 检查并加载或创建向量数据库 (Vectorstore)
# ！！！关键改动点！！！
if os.path.exists(CHROMA_DB_PATH):
    print(f"检测到现有向量数据库于 '{CHROMA_DB_PATH}'，正在加载...")
    docsearch = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    print("向量数据库加载完成。")
else:
    print(f"未检测到向量数据库于 '{CHROMA_DB_PATH}'，正在从文档创建并持久化...")
    docsearch = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_DB_PATH)
    # 调用 persist() 方法将数据库保存到磁盘
    # Chroma 1.x 版本 from_documents 可能会自动调用 persist，但显式调用更保险
    docsearch.persist()
    print("向量数据库创建并保存完成。")

print("\n--- 步骤4: 初始化语言模型 ---")
# 6. 初始化语言模型 (LLM)
llm = ChatOpenAI(
    temperature=0.7,  # 增加一点温度，让回复更自然
    openai_api_base=base_url,
    model_name="gpt-3.5-turbo",  # 确保此模型被你的第三方服务支持
    openai_api_key=api_key  # 传递API Key
)
print("语言模型初始化完成。")


# --- Chain-of-Thought (CoT) 和 RAG 逻辑实现 ---

def process_query_with_cot_and_rag(query: str, llm: ChatOpenAI, docsearch: Chroma) -> str:
    """
    结合 Chain-of-Thought (CoT) 和 Retrieval Augmented Generation (RAG) 处理用户查询。

    步骤：
    1. 情绪状态分析与初步支持 (CoT):
       - LLM 识别用户情绪。
       - LLM 提供同理心回复并建议轻度心理练习。
    2. 医疗信息需求判断 (CoT):
       - LLM 判断用户查询是否涉及需要医疗科普知识或专业指导。
    3. 医疗信息检索与安全引导 (RAG):
       - 如果判断需要医疗信息，则从知识库中检索相关内容。
       - LLM 结合检索到的信息，以科普、非诊断性语言给出回答，并强烈建议寻求专业帮助。
    """

    # --- Phase 1: 情绪状态分析与初步支持 ---
    # 定义 CoT 情绪分析和初步回复的提示模板
    emotion_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "你是一个富有同理心和支持性的AI心理健康伙伴。你的任务是帮助用户处理情绪，并提供轻度的自我关怀建议。"
            "请遵循以下步骤思考并给出你的回复：\n\n"
            "1. **识别情绪**: 分析用户消息，识别出其可能的情绪状态（如焦虑、悲伤、沮丧、平静、困惑等）。\n"
            "2. **表达同理心与安慰**: 根据识别出的情绪，用温暖、理解的语言表达同理心和安慰。\n"
            "3. **提供建议**: 提供一个简短、易于执行的轻度心理自我关怀建议或练习（例如：深呼吸、写日记、放松、小憩、散步等）。\n\n"
            "请以以下格式输出你的回复，不要包含步骤说明：\n"
            "情绪: [识别出的情绪]\n"
            "回复: [同理心和安慰的话语]\n"
            "建议: [具体的自我关怀建议]"
        ),
        HumanMessagePromptTemplate.from_template("{query}")
    ])

    emotion_chain = emotion_prompt_template | llm

    # 模拟 LLM 的情绪分析和初步回复
    try:
        emotion_response_raw = emotion_chain.invoke({"query": query}).content
        # 尝试解析 LLM 输出，虽然 LLM 可能不完全按照格式输出，但在真实场景下，可以加更强的解析

        # 简单的解析，假设LLM会遵循格式
        emotion_line = next((line for line in emotion_response_raw.split('\n') if line.startswith('情绪:')),
                            '情绪: 未知')
        reply_line = next((line for line in emotion_response_raw.split('\n') if line.startswith('回复:')),
                          '回复: 抱歉，我未能完全理解您的情绪。')
        suggestion_line = next((line for line in emotion_response_raw.split('\n') if line.startswith('建议:')),
                               '建议: 您可以尝试放松一下。')

        identified_emotion = emotion_line.replace('情绪:', '').strip()
        initial_reply = reply_line.replace('回复:', '').strip()
        self_care_suggestion = suggestion_line.replace('建议:', '').strip()

        print(f"\n--- AI 情绪分析与初步回复 ---")
        print(f"情绪: {identified_emotion}")
        print(f"回复: {initial_reply}")
        print(f"建议: {self_care_suggestion}")

    except Exception as e:
        print(f"情绪分析阶段发生错误: {e}")
        identified_emotion = "未知"
        initial_reply = "抱歉，我暂时无法处理您的情绪。或许我们可以换个话题？"
        self_care_suggestion = "保持积极心态。"
        emotion_response_raw = f"情绪: 未知\n回复: {initial_reply}\n建议: {self_care_suggestion}"

    # --- Phase 2: 医疗信息需求判断 ---
    medical_need_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "用户正在与一个心理健康AI伙伴对话。你的任务是判断用户当前的提问是否可能涉及更深层的心理健康问题、疾病诊断、药物治疗、或者需要专业医疗机构的介入。"
            "如果用户仅仅是表达情绪、寻求日常安慰或自我关怀建议，则不视为需要医疗信息。\n"
            "请只回答 '是' 或 '否'。不包含任何其他文字。"
        ),
        HumanMessagePromptTemplate.from_template("用户的问题是: '{query}'")
    ])

    medical_need_chain = medical_need_prompt_template | llm

    print(f"\n--- AI 医疗信息需求判断 ---")
    medical_need_decision = medical_need_chain.invoke({"query": query}).content.strip().lower()
    print(f"判断结果: {medical_need_decision}")

    final_response = ""

    # --- Phase 3: 医疗信息检索与安全引导 (RAG) ---
    if "是" in medical_need_decision:
        print("\n--- 正在检索相关医疗知识 ---")
        retrieved_docs = docsearch.as_retriever(search_kwargs={"k": 3}).invoke(query)  # 检索最相关的3个文档

        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 定义 RAG 医疗信息引导的提示模板
        medical_guidance_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "你是一个AI心理健康信息引导助手。你的任务是根据提供的背景信息，为用户提供关于心理健康的**科普知识**，而不是诊断或治疗建议。\n"
                "请严格遵守以下规则：\n"
                "1. **不要诊断**: 绝不能直接或间接诊断用户的任何疾病。\n"
                "2. **提供科普**: 仅提供与用户问题相关的通用、权威的背景信息（来自知识库）。\n"
                "3. **强调非专业性**: 每次回答后，必须明确告知用户你的信息不能替代专业医疗建议。\n"
                "4. **鼓励寻求专业帮助**: 强烈建议用户咨询注册医生、心理治疗师或精神科医生。\n"
                "5. **语气谨慎**: 保持客观、专业和谨慎的语气。\n"
                "--- 背景信息 ---\n{context}\n\n"
                "--- 用户问题 ---"
            ),
            HumanMessagePromptTemplate.from_template("{query}")
        ])

        medical_guidance_chain = medical_guidance_prompt_template | llm

        try:
            rag_response = medical_guidance_chain.invoke({"query": query, "context": context_text}).content
            final_response = (
                f"{initial_reply} {self_care_suggestion}\n\n"
                f"根据您刚才的提问，我为您整理了一些相关信息：\n"
                f"{rag_response}\n\n"
                f"**请注意：我是一个AI，不能进行诊断或提供医疗建议。我提供的信息仅供科普参考。如果您感到不适，请务必咨询专业的医生或心理健康专家。**"
            )
            print("\n--- AI 结合医疗知识的最终回复 ---")
            print(final_response)
        except Exception as e:
            final_response = (
                f"{initial_reply} {self_care_suggestion}\n\n"
                f"抱歉，在检索医疗信息时出现了一些问题。请确保您的网络连接正常，或稍后再试。\n"
                f"**请注意：我是一个AI，不能进行诊断或提供医疗建议。如果您感到不适，请务必咨询专业的医生或心理健康专家。**"
            )
            print(f"医疗信息检索阶段发生错误: {e}")
            print(final_response)


    else:
        # 如果不需要医疗信息，则只返回情绪分析和初步支持的回复
        final_response = (
            f"{initial_reply} {self_care_suggestion}"
        )
        print("\n--- AI 最终回复 (情绪支持) ---")
        print(final_response)

    return final_response


print("\n--- 步骤5: 开始提问 (CoT + RAG) ---")
# 7. 开始提问
while True:
    query = input("\n请输入你的问题 (输入 '退出' 结束): ")
    if query.lower() == '退出':
        print("谢谢使用，再见！")
        break

    print(f"\n正在处理您的请求: '{query}'...")
    process_query_with_cot_and_rag(query, llm, docsearch)
