from fastapi import Body, Request
from sse_starlette.sse import EventSourceResponse
from fastapi.concurrency import run_in_threadpool
from configs import (LLM_MODELS, 
                     VECTOR_SEARCH_TOP_K, 
                     SCORE_THRESHOLD, 
                     TEMPERATURE,
                     USE_RERANKER,
                     RERANKER_MODEL,
                     RERANKER_MAX_LENGTH,
                     MODEL_PATH)
from server.utils import wrap_done, get_ChatOpenAI
from server.utils import BaseResponse, get_prompt_template
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBServiceFactory
import json
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
# from server.knowledge_base.kb_doc_api import search_unrelated_docs
from server.reranker.reranker import LangchainReranker
from server.utils import embedding_device
import random

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                              top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                              score_threshold: float = Body(
                                  SCORE_THRESHOLD,
                                  description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                  ge=0,
                                  le=2
                              ),
                              history: List[History] = Body(
                                  [],
                                  description="历史对话",
                                  examples=[[
                                      {"role": "user",
                                       "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                      {"role": "assistant",
                                       "content": "虎头虎脑"}]]
                              ),
                              stream: bool = Body(False, description="流式输出"),
                              model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                              temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                              max_tokens: Optional[int] = Body(
                                  None,
                                  description="限制LLM生成Token数量，默认None代表模型最大值"
                              ),
                              prompt_name: str = Body(
                                  "default",
                                  description="使用的prompt模板名称(在configs/prompt_config.py中配置)"
                              ),
                              request: Request = None,
                              ):
    # 通过知识库工厂，检查知识库是否存在
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
    

    # 构造历史对话记录
    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator(
            query: str,
            top_k: int,
            history: Optional[List[History]],
            model_name: str = model_name,
            prompt_name: str = prompt_name,
    ) -> AsyncIterable[str]:
        nonlocal max_tokens
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )

               
        # 通过向量数据库，搜索相关文档
        # run_in_threadpool: 这是一个异步函数，它的作用是在线程池中执行一个函数。
        # 线程池是为了处理那些会阻塞主事件循环的操作而设计的。这样可以确保主事件循环不会被阻塞，保持了程序的响应性。
        docs = await run_in_threadpool(search_docs,
                                       query=query,
                                       knowledge_base_name=knowledge_base_name,
                                       top_k=top_k,
                                       score_threshold=score_threshold)

        # 加入reranker
        if USE_RERANKER:
            reranker_model_path = MODEL_PATH["reranker"].get(RERANKER_MODEL,"BAAI/bge-reranker-large")
            print("-----------------model path------------------")
            print(reranker_model_path)
            reranker_model = LangchainReranker(top_n=top_k,
                                            device=embedding_device(),
                                            max_length=RERANKER_MAX_LENGTH,
                                            model_name_or_path=reranker_model_path
                                            )
            # print(unrelated_docs_combined)
            # unrelated_docs_combined = reranker_model.compress_documents(documents=unrelated_docs_combined,
            #                                                             query=query)
            print("---------before rerank------------------")
            # print(unrelated_docs_combined)
            print(docs)
            # # reranker_model.compress_documents 函数的工作方式是用来重新排序从知识库检索的文档。
            # # 它接收文档和查询作为参数，并根据查询的相关性返回新的文档顺序。这个函数在 knowledge_base_chat_iterator 函数中的目的是提高用于生成聊天响应的文档的质量。
            # # 通过重新排序文档，该函数确保首先使用最相关的文档，这可以导致更准确和有用的响应。
            docs = reranker_model.compress_documents(documents=docs,
                                                     query=query)
            print("---------after rerank------------------")
            print(docs)

        # # 根据问题（Q）搜索与之无关的条目
        # unrelated_docs_q = await run_in_threadpool(search_unrelated_docs,
        #                                            query=query,
        #                                            knowledge_base_name=knowledge_base_name,
        #                                            top_k=top_k,
        #                                            score_threshold=score_threshold)
        
        # # 根据上下文（C）搜索与之无关的条目
        # context = "\n".join([doc.page_content for doc in history])
        # unrelated_docs_c = await run_in_threadpool(search_unrelated_docs,
        #                                            query=context,
        #                                            knowledge_base_name=knowledge_base_name,
        #                                            top_k=top_k,
        #                                            score_threshold=score_threshold)

        # # 合并无关的条目和原始上下文（C）
        # unrelated_docs_combined = random.sample(unrelated_docs_q, k=top_k) + random.sample(unrelated_docs_c, k=top_k)
        # new_context = "\n".join([doc.page_content for doc in unrelated_docs_combined]) + "\n" + "\n".join([doc.page_content for doc in docs])
        # context=new_context


        unrelated_docs_query = ' '.join([doc.page_content for doc in docs])  # 使用所有文档内容作为查询
        unrelated_docs = await run_in_threadpool(search_docs,
                                                 query=unrelated_docs_query,
                                                 knowledge_base_name=knowledge_base_name,
                                                 top_k=top_k,
                                                 score_threshold=score_threshold)
                
        # 将查询到的与文档无关的条目与原始文档随机拼接成新的上下文
        combined_docs = docs + random.sample(unrelated_docs, min(len(unrelated_docs), len(docs)))  # 随机取样本
        random.shuffle(combined_docs)  # 随机打乱顺序
        combined_context = "\n".join([doc.page_content for doc in combined_docs])
        context = combined_context

        # context = "\n".join([doc.page_content for doc in docs])
        if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
            prompt_template = get_prompt_template("knowledge_base_chat", "empty")
        else:
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        
        # 构造chain对象
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # 通过chain对象异步调用大模型
        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )
        # 原文出处
        source_documents = []
        for inum, doc in enumerate(docs):
            filename = doc.metadata.get("source")
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
            base_url = request.base_url
            url = f"{base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if len(source_documents) == 0:  # 没有找到相关文档
            source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)
        await task

    return EventSourceResponse(knowledge_base_chat_iterator(query, top_k, history,model_name,prompt_name))

