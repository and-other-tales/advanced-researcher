"""Chain implementation for local deployment."""
import asyncio
import os
from operator import itemgetter
from typing import Dict, List, Optional, Sequence, Any

from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableSequence,
    chain,
)
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langsmith import Client

from backend.local_embeddings import get_embeddings_model
from backend.dynamic_chain import ChatRequestWithKB, create_dynamic_chain
from backend.auto_learn import detect_insufficient_information, register_learning_task
from backend.verification import get_verification_components, VerificationComponents

RESPONSE_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Langchain.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""


client = Client()


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None
    knowledge_base_id: Optional[str] = None
    advanced_verification: Optional[bool] = False


def get_retriever() -> BaseRetriever:
    """Get the retriever to use."""
    # Use local environment variables to decide on vector store
    if os.environ.get("WEAVIATE_URL") and os.environ.get("WEAVIATE_API_KEY"):
        # Use Weaviate if configured
        import weaviate
        from langchain_community.vectorstores import Weaviate
        from backend.constants import WEAVIATE_DOCS_INDEX_NAME
        
        weaviate_client = weaviate.Client(
            url=os.environ["WEAVIATE_URL"],
            auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"]),
        )
        weaviate_client = Weaviate(
            client=weaviate_client,
            index_name=WEAVIATE_DOCS_INDEX_NAME,
            text_key="text",
            embedding=get_embeddings_model(),
            by_text=False,
            attributes=["source", "title"],
        )
        return weaviate_client.as_retriever(search_kwargs=dict(k=6))
    else:
        # Use Chroma for local development
        collection_name = os.environ.get("COLLECTION_NAME", "langchain")
        chroma_client = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings_model(),
        )
        return chroma_client.as_retriever(search_kwargs=dict(k=6))


def create_retriever_chain(
    llm: LanguageModelLike, retriever: BaseRetriever
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    # Create standard verification components (advanced verification will be enabled per request)
    verification = get_verification_components(llm, advanced_verification=False)
    
    # Retrieval chain with document relevance filtering
    def retrieve_and_filter_docs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Check if advanced verification is enabled for this request
        advanced_verification = inputs.get("advanced_verification", False)
        
        # Create appropriate verification components
        current_verification = get_verification_components(llm, advanced_verification=advanced_verification)
        
        # First, use the standard retriever chain
        retriever_result = create_retriever_chain(llm, retriever).invoke(inputs)
        
        # Then filter for relevance using verification
        question = inputs.get("question", "")
        filtered_docs = current_verification.filter_relevant_documents(question, retriever_result)
        
        return {**inputs, "docs": filtered_docs, "advanced_verification": advanced_verification}
    
    # Enhanced context preparation
    context = (
        RunnableLambda(retrieve_and_filter_docs).with_config(run_name="RetrieveAndFilterDocs")
        .assign(context=lambda x: format_docs(x["docs"]))
        .with_config(run_name="PrepareContext")
    )
    
    # Answer generation
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    
    response_synthesizer = (
        prompt | llm | StrOutputParser()
    ).with_config(run_name="GenerateResponse")
    
    # Verification and improvement of generated answer
    def verify_and_improve_answer(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs.get("question", "")
        docs = inputs.get("docs", [])
        generated_answer = inputs.get("generated_answer", "")
        advanced_verification = inputs.get("advanced_verification", False)
        
        # Create appropriate verification components for this request
        current_verification = get_verification_components(llm, advanced_verification=advanced_verification)
        
        # Run verification checks
        verification_result = current_verification.verify_and_improve_answer(
            question, generated_answer, docs
        )
        
        # Use the verified (potentially improved) answer
        return {
            **inputs, 
            "response": verification_result["verified_answer"],
            "verification_result": verification_result,
            "advanced_verification": advanced_verification
        }
    
    # Add auto-learning capability when information is insufficient
    def check_and_learn(inputs: Dict[str, Any]) -> Dict[str, Any]:
        response = inputs.get("response", "")
        question = inputs.get("question", "")
        verification_result = inputs.get("verification_result", {})
        
        # Check if verification indicates insufficient information or hallucinations
        needs_learning = False
        
        # If verification shows hallucinations or missing information
        if verification_result.get("needs_improvement", False):
            needs_learning = True
        
        # Also check with the insufficient information detector
        if detect_insufficient_information(question, response):
            needs_learning = True
        
        if needs_learning:
            try:
                # Trigger background learning process
                asyncio.create_task(register_learning_task(question, llm))
                
                # Add learning notification to the response
                learning_msg = "\n\nI'm searching for more information on this topic and will update my knowledge base."
                return {**inputs, "response": response + learning_msg}
            except Exception as e:
                # If learning fails, just return the original response
                return inputs
        
        return inputs
    
    # Combine everything into the final chain
    chain = (
        RunnablePassthrough.assign(chat_history=serialize_history)
        | context
        | RunnablePassthrough.assign(generated_answer=response_synthesizer)
        | RunnableLambda(verify_and_improve_answer).with_config(run_name="VerifyAndImprove")
    )
    
    # Add learning check to the chain
    return (
        chain
        | RunnableLambda(check_and_learn).with_config(run_name="CheckAndLearn")
        | itemgetter("response")
    )


def get_llm() -> LanguageModelLike:
    """Get the language model to use."""
    # Use Ollama for local LLM if specified
    if os.environ.get("USE_OLLAMA") == "true":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model="mistral",
            base_url=base_url,
            temperature=0,
        )
    
    # Default to OpenAI
    return ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        temperature=0,
        streaming=True,
    )


# Initialize the base chain
llm = get_llm()
retriever = get_retriever()
base_chain = create_chain(llm, retriever)

# Create the dynamic chain that supports knowledge base selection
answer_chain = create_dynamic_chain(base_chain, llm)