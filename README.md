# LangChain 中文入门教程

> 为了便于阅读，已生成 [gitbook](https://liaokong.gitbook.io/llm-kai-fa-jiao-cheng/)
>
> [项目原始地址](https://github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide)

> 加了个 [CHANGELOG](CHANGELOG.md),更新了新的内容我会写在这里，方便之前看过的朋友快速查看新的更新内容

> 如果想把 OPENAI API 的请求根路由修改成自己的代理地址，可以通过设置环境变量 “OPENAI\_API\_BASE” 来进行修改。
>
> 相关参考代码可以参考[这里](https://github.com/openai/openai-python/blob/d6fa3bfaae69d639b0dd2e9251b375d7070bbef1/openai/\_\_init\_\_.py#L48)
>
> 或在初始化OpenAI相关模型对象时，传入“openai\_api\_base” 变量。
>
> 相关代码参考[这里](https://github.com/hwchase17/langchain/blob/master/langchain/llms/openai.py#L148)
> 
> <span style="color: rgb(255, 102, 0);">上面的 2 个变量名称是没有错误的，一个大写一个小写，请务必注意</span>

## 介绍

众所周知 OpenAI 的 API 无法联网的（2023 年 5 月份已经推出了插件服务，这里指的是早于这个时间），所以如果只使用自己的功能实现联网搜索并给出回答、总结 PDF 文档、基于某个 Youtube 视频进行问答等等的功能肯定是无法实现的。

所以，我们来介绍一个非常强大的第三方开源库：`LangChain` ，项目地址在[这里](https://github.com/hwchase17/langchain)，文档地址在[这里](https://python.langchain.com/en/latest/)。

这个库目前非常活跃，每天都在迭代，已经有 ~22k~ 44K 的 star，更新速度飞快。

LangChain 是一个用于开发由语言模型驱动的应用程序的框架。他主要拥有 2 个能力：

1. 可以将 LLM 模型与外部数据源进行连接
2. 允许与 LLM 模型进行交互

> LLM 模型：Large Language Model，大型语言模型


## 基础功能

LLM 调用

* 支持多种模型接口，比如 OpenAI、Hugging Face、AzureOpenAI ...
* Fake LLM，用于测试
* 缓存的支持，比如 in-mem（内存）、SQLite、Redis、SQL
* 用量记录
* 支持流模式（就是一个字一个字的返回，类似打字效果）

Prompt管理，支持各种自定义模板

拥有大量的文档加载器，比如 Email、Markdown、PDF、Youtube ...

对索引的支持

* 文档分割器
* 向量化
* 对接向量存储与搜索，比如 Chroma、Pinecone、Qdrand

Chains

* LLMChain
* 各种工具 Chain
* LangChainHub

## 必知概念

上面的概念其实在刚开始学的时候不是很重要，当我们讲完后面的例子之后，在回来看上面的内容会一下明白很多。

但是，这里有几个概念是必须知道的。

### Loader 加载器

顾名思义，这个就是从指定源进行加载数据的。比如：

* 文件夹 `DirectoryLoader`、
* Azure 存储 `AzureBlobStorageContainerLoader`、
* CSV文件 `CSVLoader`、
* 印象笔记 `EverNoteLoader`、
* Google网盘 `GoogleDriveLoader`、
* 任意的网页 `UnstructuredHTMLLoader`、
* PDF `PyPDFLoader`、
* S3 `S3DirectoryLoader`/`S3FileLoader`、
* Youtube `YoutubeLoader` 

等等，上面只是简单的进行列举了几个，官方提供了[超级多的加载器](https://python.langchain.com/en/latest/modules/indexes/document\_loaders.html)供你使用。

### Document 文档

当使用loader加载器读取到数据源后，数据源需要转换成 Document 对象后，后续才能进行使用。

### Text Splitters 文本分割

顾名思义，文本分割就是用来分割文本的。

为什么需要分割文本？因为我们每次不管是做把文本当作 prompt 发给 OpenAI API，还是还是使用 OpenAI API embedding 功能都是有字符限制的。

比如我们将一份 300 页的 PDF 文档发给 OpenAI api，让他进行总结，他肯定会报`超过最大 Token` 的错误。所以这里就需要使用文本分割器去分割我们加载(load)进来的 Document。


### Vectorstores 向量数据库

因为数据相关性搜索其实是向量运算。

所以，不管我们是使用 OpenAI API embedding 功能还是直接通过向量数据库直接查询，都需要将我们的加载进来的数据 `Document` 进行向量化，才能进行向量运算搜索。

转换成向量也很简单，只需要我们把数据存储到对应的向量数据库中即可完成向量的转换。

官方也提供了很多的[向量数据库](https://python.langchain.com/en/latest/modules/indexes/vectorstores.html)供我们使用。

### Chain 链

> 有点像编程语言中的链式操作，可以一直点下去的样子。

我们可以把 Chain 理解为任务。一个 Chain 就是一个任务，当然也可以像链条一样，一个一个的执行多个链。

### Agent 代理

我们可以简单的理解为它(Agent)可以动态的帮我们选择和调用 Chain 或者已有的工具。

执行过程可以参考下面这张图:

![image-20230406213322739](doc/image-20230406213322739.png)

### Embedding

用于衡量文本的相关性。

这个也是 OpenAI API 能实现构建自己知识库的关键所在。

它相比 fine-tuning 最大的优势就是，不用进行训练，并且可以实时添加新的内容，而不用加一次新的内容就训练一次，并且各方面成本要比 fine-tuning 低很多。

具体比较和选择可以参考这个[视频](https://www.youtube.com/watch?v=9qq6HTr7Ocw)

## 实战

通过上面的必备概念大家应该已经可以对 LangChain 有了一定的了解，但是可能还有有些懵。

这都是小问题，我相信看完后面的实战，你们就会彻底的理解上面的内容，并且能感受到这个库的真正强大之处。

因为我们 OpenAI API 进阶，所以我们后面的范例使用的 LLM 都是以Open AI 为例，后面大家可以根据自己任务的需要换成自己需要的 LLM 模型即可。

当然，在这篇文章的末尾，全部的全部代码都会被保存为一个 colab 的 ipynb 文件提供给大家来学习。

> 建议大家按顺序去看每个例子，因为下一个例子会用到上一个例子里面的知识点。
>
> 当然，如果有看不懂的也不用担心，可以继续往后看，第一次学习讲究的是不求甚解。

###

### 完成一次问答

第一个案例，我们就来个最简单的，用 LangChain 加载 OpenAI 的模型，并且完成一次问答。

在开始之前，我们需要先设置我们的 openai 的 key，这个 key 可以在用户管理里面创建，这里就不细说了。

```python
import os
os.environ["OPENAI_API_KEY"] = '你的api key'
```

然后，我们进行导入和执行

```py
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003",max_tokens=1024)
llm("怎么评价人工智能")
```

![image-20230404232621517](doc/image-20230404232621517.png)

这时，我们就可以看到他给我们的返回结果了，怎么样，是不是很简单。

### 通过 Google 搜索并返回答案

接下来，我们就来搞点有意思的。我们来让我们的 OpenAI api 联网搜索，并返回答案给我们。

这里我们需要借助 Serpapi 来进行实现，Serpapi 提供了 google 搜索的 api 接口。

首先需要我们到 Serpapi 官网上注册一个用户，https://serpapi.com/ 并复制他给我们生成 api key。

然后我们需要像上面的 openai api key 一样设置到环境变量里面去。

```python
import os
os.environ["OPENAI_API_KEY"] = '你的api key'
os.environ["SERPAPI_API_KEY"] = '你的api key'
```

然后，开始编写我的代码

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import AgentType

# 加载 OpenAI 模型
llm = OpenAI(temperature=0,max_tokens=2048) 

 # 加载 serpapi 工具
tools = load_tools(["serpapi"])

# 如果搜索完想在计算一下可以这么写
# tools = load_tools(['serpapi', 'llm-math'], llm=llm)

# 如果搜索完想再让他再用python的print做点简单的计算，可以这样写
# tools=load_tools(["serpapi","python_repl"])

# 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 运行 agent
agent.run("What's the date today? What great events have taken place today in history?")
```

![image-20230404234236982](doc/image-20230404234236982.png)

我们可以看到，他正确的返回了日期（有时差），并且返回了历史上的今天。

在 chain 和 agent 对象上都会有 `verbose` 这个参数，这个是个非常有用的参数，开启他后我们可以看到完整的 chain 执行过程。

可以在上面返回的结果看到，他将我们的问题拆分成了几个步骤，然后一步一步得到最终的答案。

关于agent type 几个选项的含义（理解不了也不会影响下面的学习，用多了自然理解了）：

* zero-shot-react-description: 根据工具的描述和请求内容的来决定使用哪个工具（最常用）
* react-docstore: 使用 ReAct 框架和 docstore 交互, 使用`Search` 和`Lookup` 工具, 前者用来搜, 后者寻找term, 举例: `Wipipedia` 工具
* self-ask-with-search 此代理只使用一个工具: Intermediate Answer, 它会为问题寻找事实答案(指的非 gpt 生成的答案, 而是在网络中,文本中已存在的), 如 `Google search API` 工具
* conversational-react-description: 为会话设置而设计的代理, 它的prompt会被设计的具有会话性, 且还是会使用 ReAct 框架来决定使用来个工具, 并且将过往的会话交互存入内存

> reAct 介绍可以看这个：https://arxiv.org/pdf/2210.03629.pdf
>
> LLM 的 ReAct 模式的 Python 实现: https://til.simonwillison.net/llms/python-react-pattern
>
> agent type 官方解释：
>
> https://python.langchain.com/en/latest/modules/agents/agents/agent\_types.html?highlight=zero-shot-react-description

> 有一点要说明的是，这个 `serpapi` 貌似对中文不是很友好，所以提问的 prompt 建议使用英文。

当然，官方已经写好了 `ChatGPT Plugins` 的 agent，未来 chatgpt 能用啥插件，我们在 api 里面也能用插件，想想都美滋滋。

不过目前只能使用不用授权的插件，期待未来官方解决这个。

感兴趣的可以看这个文档：https://python.langchain.com/en/latest/modules/agents/tools/examples/chatgpt\_plugins.html

> Chatgpt 只能给官方赚钱，而 Openai API 能给我赚钱

### 对超长文本进行总结

假如我们想要用 openai api 对一个段文本进行总结，我们通常的做法就是直接发给 api 让他总结。但是如果文本超过了 api 最大的 token 限制就会报错。

这时，我们一般会进行对文章进行分段，比如通过 tiktoken 计算并分割，然后将各段发送给 api 进行总结，最后将各段的总结再进行一个全部的总结。

如果，你用是 LangChain，他很好的帮我们处理了这个过程，使得我们编写代码变的非常简单。

废话不多说，直接上代码。

```python
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI

# 导入文本
loader = UnstructuredFileLoader("/content/sample_data/data/lg_test.txt")
# 将文本转成 Document 对象
document = loader.load()
print(f'documents:{len(document)}')

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 0
)

# 切分文本
split_documents = text_splitter.split_documents(document)
print(f'documents:{len(split_documents)}')

# 加载 llm 模型
llm = OpenAI(model_name="text-davinci-003", max_tokens=1500)

# 创建总结链
chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

# 执行总结链，（为了快速演示，只总结前5段）
chain.run(split_documents[:5])
```

首先我们对切割前和切割后的 document 个数进行了打印，我们可以看到，切割前就是只有整篇的一个 document，切割完成后，会把上面一个 document 切成 317 个 document。

![image-20230405162631460](doc/image-20230405162631460.png)

最终输出了对前 5 个 document 的总结。

![image-20230405162937249](doc/image-20230405162937249.png)

这里有几个参数需要注意：

**文本分割器的 `chunk_overlap` 参数**

这个是指切割后的每个 document 里包含几个上一个 document 结尾的内容，主要作用是为了增加每个 document 的上下文关联。比如，`chunk_overlap=0`时， 第一个 document 为 aaaaaa，第二个为 bbbbbb；当 `chunk_overlap=2` 时，第一个 document 为 aaaaaa，第二个为 aabbbbbb。

不过，这个也不是绝对的，要看所使用的那个文本分割模型内部的具体算法。

> 文本分割器可以参考这个文档：https://python.langchain.com/en/latest/modules/indexes/text\_splitters.html

**chain 的 `chain_type` 参数**

这个参数主要控制了将 document 传递给 llm 模型的方式，一共有 4 种方式：

`stuff`: 这种最简单粗暴，会把所有的 document 一次全部传给 llm 模型进行总结。如果document很多的话，势必会报超出最大 token 限制的错，所以总结文本的时候一般不会选中这个。

`map_reduce`: 这个方式会先将每个 document 进行总结，最后将所有 document 总结出的结果再进行一次总结。

![image-20230405165752743](doc/image-20230405165752743.png)

`refine`: 这种方式会先总结第一个 document，然后在将第一个 document 总结出的内容和第二个 document 一起发给 llm 模型在进行总结，以此类推。这种方式的好处就是在总结后一个 document 的时候，会带着前一个的 document 进行总结，给需要总结的 document 添加了上下文，增加了总结内容的连贯性。

![image-20230405170617383](doc/image-20230405170617383.png)

`map_rerank`: 这种一般不会用在总结的 chain 上，而是会用在问答的 chain 上，他其实是一种搜索答案的匹配方式。首先你要给出一个问题，他会根据问题给每个 document 计算一个这个 document 能回答这个问题的概率分数，然后找到分数最高的那个 document ，在通过把这个 document 转化为问题的 prompt 的一部分（问题+document）发送给 llm 模型，最后 llm 模型返回具体答案。

### 构建本地知识库问答机器人

在这个例子中，我们会介绍如何从我们本地读取多个文档构建知识库，并且使用 Openai API 在知识库中进行搜索并给出答案。

这个是个很有用的教程，比如可以很方便的做一个可以介绍公司业务的机器人，或是介绍一个产品的机器人。

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI,VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA

# 加载文件夹中的所有txt类型的文件
loader = DirectoryLoader('/content/sample_data/data/', glob='**/*.txt')
# 将数据转成 document 对象，每个文件会作为一个 document
documents = loader.load()

# 初始化加载器
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# 切割加载的 document
split_docs = text_splitter.split_documents(documents)

# 初始化 openai 的 embeddings 对象
embeddings = OpenAIEmbeddings()
# 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
docsearch = Chroma.from_documents(split_docs, embeddings)

# 创建问答对象
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch,return_source_documents=True)
# 进行问答
result = qa({"query": "科大讯飞今年第一季度收入是多少？"})
print(result)
```

![image-20230405173730382](doc/image-20230405173730382.png)

我们可以通过结果看到，他成功的从我们的给到的数据中获取了正确的答案。

> 关于 Openai embeddings 详细资料可以参看这个连接: https://platform.openai.com/docs/guides/embeddings

### 构建向量索引数据库

我们上个案例里面有一步是将 document 信息转换成向量信息和embeddings的信息并临时存入 Chroma 数据库。

因为是临时存入，所以当我们上面的代码执行完成后，上面的向量化后的数据将会丢失。如果想下次使用，那么就还需要再计算一次embeddings，这肯定不是我们想要的。

那么，这个案例我们就来通过 Chroma 和 Pinecone 这两个数据库来讲一下如何做向量数据持久化。

> 因为 LangChain 支持的数据库有很多，所以这里就介绍两个用的比较多的，更多的可以参看文档:https://python.langchain.com/en/latest/modules/indexes/vectorstores/getting\_started.html

**Chroma**

chroma 是个本地的向量数据库，他提供的一个 `persist_directory` 来设置持久化目录进行持久化。读取时，只需要调取 `from_document` 方法加载即可。

```python
from langchain.vectorstores import Chroma

# 持久化数据
docsearch = Chroma.from_documents(documents, embeddings, persist_directory="D:/vector_store")
docsearch.persist()

# 加载数据
docsearch = Chroma(persist_directory="D:/vector_store", embedding_function=embeddings)

```

**Pinecone**

Pinecone 是一个在线的向量数据库。所以，我可以第一步依旧是注册，然后拿到对应的 api key。https://app.pinecone.io/

> 免费版如果索引14天不使用会被自动清除。

然后创建我们的数据库：

Index Name：这个随意

Dimensions：OpenAI 的 text-embedding-ada-002 模型为 OUTPUT DIMENSIONS 为 1536，所以我们这里填 1536

Metric：可以默认为 cosine

选择starter plan

![image-20230405184646314](doc/starter-plan.png)

持久化数据和加载数据代码如下

```python
# 持久化数据
docsearch = Pinecone.from_texts([t.page_content for t in split_docs], embeddings, index_name=index_name)

# 加载数据
docsearch = Pinecone.from_existing_index(index_name, embeddings)
```

一个简单从数据库获取 embeddings，并回答的代码如下

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import pinecone

# 初始化 pinecone
pinecone.init(
  api_key="你的api key",
  environment="你的Environment"
)

loader = DirectoryLoader('/content/sample_data/data/', glob='**/*.txt')
# 将数据转成 document 对象，每个文件会作为一个 document
documents = loader.load()

# 初始化加载器
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# 切割加载的 document
split_docs = text_splitter.split_documents(documents)

index_name="liaokong-test"

# 持久化数据
# docsearch = Pinecone.from_texts([t.page_content for t in split_docs], embeddings, index_name=index_name)

# 加载数据
docsearch = Pinecone.from_existing_index(index_name,embeddings)

query = "科大讯飞今年第一季度收入是多少？"
docs = docsearch.similarity_search(query, include_metadata=True)

llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
chain.run(input_documents=docs, question=query)
```

![image-20230407001803057](doc/image-20230407001803057.png)

### 使用GPT3.5模型构建油管频道问答机器人

在 chatgpt api（也就是 GPT-3.5-Turbo）模型出来后，因钱少活好深受大家喜爱，所以 LangChain 也加入了专属的链和模型，我们来跟着这个例子看下如何使用他。

```python
import os

from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate
)

# 加载 youtube 频道
loader = YoutubeLoader.from_youtube_url('https://www.youtube.com/watch?v=Dj60HHy-Kqk')
# 将数据转成 document
documents = loader.load()

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap=20
)

# 分割 youtube documents
documents = text_splitter.split_documents(documents)

# 初始化 openai embeddings
embeddings = OpenAIEmbeddings()

# 将数据存入向量存储
vector_store = Chroma.from_documents(documents, embeddings)
# 通过向量存储初始化检索器
retriever = vector_store.as_retriever()

system_template = """
Use the following context to answer the user's question.
If you don't know the answer, say you don't, don't try to make it up. And answer in Chinese.
-----------
{context}
-----------
{chat_history}
"""

# 构建初始 messages 列表，这里可以理解为是 openai 传入的 messages 参数
messages = [
  SystemMessagePromptTemplate.from_template(system_template),
  HumanMessagePromptTemplate.from_template('{question}')
]

# 初始化 prompt 对象
prompt = ChatPromptTemplate.from_messages(messages)


# 初始化问答链
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.1,max_tokens=2048),retriever,condense_question_prompt=prompt)


chat_history = []
while True:
  question = input('问题：')
  # 开始发送问题 chat_history 为必须参数,用于存储对话历史
  result = qa({'question': question, 'chat_history': chat_history})
  chat_history.append((question, result['answer']))
  print(result['answer'])
```

我们可以看到他能很准确的围绕这个油管视频进行问答

![image-20230406211923672](doc/image-20230406211923672.png)

使用流式回答也很方便

```python
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

chat = ChatOpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
resp = chat(chat_prompt_with_values.to_messages())
```

### 用 OpenAI 连接万种工具

我们主要是结合使用 `zapier` 来实现将万种工具连接起来。

所以我们第一步依旧是需要申请账号和他的自然语言 api key。https://zapier.com/l/natural-language-actions

他的 api key 虽然需要填写信息申请。但是基本填入信息后，基本可以秒在邮箱里看到审核通过的邮件。

然后，我们通过右键里面的连接打开我们的api 配置页面。我们点击右侧的 `Manage Actions` 来配置我们要使用哪些应用。

我在这里配置了 Gmail 读取和发邮件的 action，并且所有字段都选的是通过 AI 猜。

![image-20230406233319250](doc/image-20230406233319250.png)

![image-20230406234827815](doc/image-20230406234827815.png)

配置好后，我们开始写代码

```python
import os
os.environ["ZAPIER_NLA_API_KEY"] = ''
```

```python
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper


llm = OpenAI(temperature=.3)
zapier = ZapierNLAWrapper()
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
agent = initialize_agent(toolkit.get_tools(), llm, agent="zero-shot-react-description", verbose=True)

# 我们可以通过打印的方式看到我们都在 Zapier 里面配置了哪些可以用的工具
for tool in toolkit.get_tools():
  print (tool.name)
  print (tool.description)
  print ("\n\n")

agent.run('请用中文总结最后一封"******@qq.com"发给我的邮件。并将总结发送给"******@qq.com"')
```

![image-20230406234712909](doc/image-20230406234712909.png)

我们可以看到他成功读取了`******@qq.com`给他发送的最后一封邮件，并将总结的内容又发送给了`******@qq.com`

这是我发送给 Gmail 的邮件。

![image-20230406234017369](doc/image-20230406234017369.png)

这是他发送给 QQ 邮箱的邮件。

![image-20230406234800632](doc/image-20230406234800632.png)

这只是个小例子，因为 `zapier` 有数以千计的应用，所以我们可以轻松结合 openai api 搭建自己的工作流。

## 小例子们

一些比较大的知识点都已经讲完了，后面的内容都是一些比较有趣的小例子，当作拓展延伸。

### **执行多个chain**

因为他是链式的，所以他也可以按顺序依次去执行多个 chain

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

# location 链
llm = OpenAI(temperature=1)
template = """Your job is to come up with a classic dish from the area that the users suggests.
% USER LOCATION
{user_location}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_location"], template=template)
location_chain = LLMChain(llm=llm, prompt=prompt_template)

# meal 链
template = """Given a meal, give a short and simple recipe on how to make that dish at home.
% MEAL
{user_meal}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_meal"], template=template)
meal_chain = LLMChain(llm=llm, prompt=prompt_template)

# 通过 SimpleSequentialChain 串联起来，第一个答案会被替换第二个中的user_meal，然后再进行询问
overall_chain = SimpleSequentialChain(chains=[location_chain, meal_chain], verbose=True)
review = overall_chain.run("Rome")
```

![image-20230406000133339](doc/image-20230406000133339.png)

### **结构化输出**

有时候我们希望输出的内容不是文本，而是像 json 那样结构化的数据。

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003")

# 告诉他我们生成的内容需要哪些字段，每个字段类型式啥
response_schemas = [
    ResponseSchema(name="bad_string", description="This a poorly formatted user input string"),
    ResponseSchema(name="good_string", description="This is your response, a reformatted response")
]

# 初始化解析器
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 生成的格式提示符
# {
#	"bad_string": string  // This a poorly formatted user input string
#	"good_string": string  // This is your response, a reformatted response
#}
format_instructions = output_parser.get_format_instructions()

template = """
You will be given a poorly formatted string from a user.
Reformat it and make sure all the words are spelled correctly

{format_instructions}

% USER INPUT:
{user_input}

YOUR RESPONSE:
"""

# 讲我们的格式描述嵌入到 prompt 中去，告诉 llm 我们需要他输出什么样格式的内容
prompt = PromptTemplate(
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions},
    template=template
)

promptValue = prompt.format(user_input="welcom to califonya!")
llm_output = llm(promptValue)

# 使用解析器进行解析生成的内容
output_parser.parse(llm_output)
```

![image-20230406000017276](doc/image-20230406000017276.png)

### **爬取网页并输出JSON数据**

有些时候我们需要爬取一些<mark style="color:red;">**结构性比较强**</mark>的网页，并且需要将网页中的信息以JSON的方式返回回来。

我们就可以使用 `LLMRequestsChain` 类去实现，具体可以参考下面代码

> 为了方便理解，我在例子中直接使用了Prompt的方法去格式化输出结果，而没用使用上个案例中用到的 `StructuredOutputParser`去格式化，也算是提供了另外一种格式化的思路

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMRequestsChain, LLMChain

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = """在 >>> 和 <<< 之间是网页的返回的HTML内容。
网页是新浪财经A股上市公司的公司简介。
请抽取参数请求的信息。

>>> {requests_result} <<<
请使用如下的JSON格式返回数据
{{
  "company_name":"a",
  "company_english_name":"b",
  "issue_price":"c",
  "date_of_establishment":"d",
  "registered_capital":"e",
  "office_address":"f",
  "Company_profile":"g"

}}
Extracted:"""

prompt = PromptTemplate(
    input_variables=["requests_result"],
    template=template
)

chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=prompt))
inputs = {
  "url": "https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpInfo/stockid/600519.phtml"
}

response = chain(inputs)
print(response['output'])
```

我们可以看到，他很好的将格式化后的结果输出了出来

<figure><img src="doc/image-20230510234934.png" alt=""><figcaption></figcaption></figure>

### **自定义agent中所使用的工具**

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper

llm = OpenAI(temperature=0)

# 初始化搜索链和计算链
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

# 创建一个功能列表，指明这个 agent 里面都有哪些可用工具，agent 执行过程可以看必知概念里的 Agent 那张图
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    )
]

# 初始化 agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 执行 agent
agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")

```

![image-20230406002117283](doc/image-20230406002117283.png)

自定义工具里面有个比较有意思的地方，使用哪个工具的权重是靠 `工具中描述内容` 来实现的，和我们之前编程靠数值来控制权重完全不同。

比如 Calculator 在描述里面写到，如果你问关于数学的问题就用他这个工具。我们就可以在上面的执行过程中看到，他在我们请求的 prompt 中数学的部分，就选用了Calculator 这个工具进行计算。

### **使用Memory实现一个带记忆的对话机器人**

上一个例子我们使用的是通过自定义一个列表来存储对话的方式来保存历史的。

当然，你也可以使用自带的 memory 对象来实现这一点。

```python
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0)

# 初始化 MessageHistory 对象
history = ChatMessageHistory()

# 给 MessageHistory 对象添加对话内容
history.add_ai_message("你好！")
history.add_user_message("中国的首都是哪里？")

# 执行对话
ai_response = chat(history.messages)
print(ai_response)
```

### **使用 Hugging Face 模型**

使用 Hugging Face 模型之前，需要先设置环境变量

```python
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''
```

使用在线的 Hugging Face 模型

```python
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":64})
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
print(llm_chain.run(question))
```

将 Hugging Face 模型直接拉到本地使用

```python
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

model_id = 'google/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True)

pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=100
)

local_llm = HuggingFacePipeline(pipeline=pipe)
print(local_llm('What is the capital of France? '))


llm_chain = LLMChain(prompt=prompt,  llm=local_llm)
question = "What is the capital of England?"
print(llm_chain.run(question))
```

将模型拉到本地使用的好处：

* 训练模型
* 可以使用本地的 GPU
* 有些模型无法在 Hugging Face 运行

### **通过自然语言执行SQL命令**

我们通过 `SQLDatabaseToolkit` 或者 `SQLDatabaseChain` 都可以实现执行SQL命令的操作

```python
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI

db = SQLDatabase.from_uri("sqlite:///../notebooks/Chinook.db")
toolkit = SQLDatabaseToolkit(db=db)

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True
)

agent_executor.run("Describe the playlisttrack table")
```

```python
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain

db = SQLDatabase.from_uri("mysql+pymysql://root:root@127.0.0.1/chinook")
llm = OpenAI(temperature=0)

db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
db_chain.run("How many employees are there?")
```

这里可以参考这两篇文档：

[https://python.langchain.com/en/latest/modules/agents/toolkits/examples/sql\_database.html](https://python.langchain.com/en/latest/modules/agents/toolkits/examples/sql\_database.html)

[https://python.langchain.com/en/latest/modules/chains/examples/sqlite.html](https://python.langchain.com/en/latest/modules/chains/examples/sqlite.html)

## 总结

所有的案例都基本已经结束了，希望大家能通过这篇文章的学习有所收获。这篇文章只是对 LangChain 一个初级的讲解，高级的功能希望大家继续探索。

并且因为 LangChain 迭代极快，所以后面肯定会随着AI继续的发展，还会迭代出更好用的功能，所以我非常看好这个开源库。

希望大家能结合 LangChain 开发出更有创意的产品，而不仅仅只搞一堆各种一键搭建chatgpt聊天客户端的那种产品。

这篇标题后面加了个 `01` 是我希望这篇文章只是一个开始，后面如何出现了更好的技术我还是希望能继续更新下去这个系列。

本文章的所有范例代码都在这里，祝大家学习愉快。

https://colab.research.google.com/drive/1ArRVMiS-YkhUlobHrU6BeS8fF57UeaPQ?usp=sharing
