import logging
import os
import sys
import tempfile
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.base import Chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.schema import BaseRetriever, Document
from script.util import load_document, MEMORY

class ConversationChainConfigurator:
    def __init__(self, llm_model_name, llm_temperature, llm_streaming, max_tokens_limit):
        self.llm = ChatOpenAI(
            model_name=llm_model_name, temperature=llm_temperature, streaming=llm_streaming
        )
        self.max_tokens_limit = max_tokens_limit

    def configure_retriever(self, docs: list[Document]) -> BaseRetriever:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
        splits = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()

        vectordb = Qdrant.from_documents(
            docs,
            embeddings,
            path='new_embeding',
            collection_name="contract_documents",
            force_recreate=True,
        )

        retriever = vectordb.as_retriever(
            search_type="mmr", search_kwargs={
                'k': 5,
                'fetch_k': 7
            },
        )

        return retriever

    def configure_chain(self, retriever: BaseRetriever) -> Chain:
        params = dict(
            llm=self.llm,
            retriever=retriever,
            memory=MEMORY,
            verbose=True,
            max_tokens_limit=self.max_tokens_limit,
        )

        return ConversationalRetrievalChain.from_llm(**params)

    def retrieval_chain(self, uploaded_files) -> Chain:
        docs = []
        temp_dir = tempfile.TemporaryDirectory()
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir.name, file.filename)
            file.save(temp_filepath)
            docs.extend(load_document(temp_filepath))

        retriever = self.configure_retriever(docs=docs)
        chain = self.configure_chain(retriever=retriever)
        return chain

def main():
    logging.basicConfig(encoding="utf-8", level=logging.INFO)
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    configurator = ConversationChainConfigurator(
        llm_model_name="gpt-3.5-turbo",
        llm_temperature=0,
        llm_streaming=True,
        max_tokens_limit=4000
    )

    chain = configurator.configure_retrieval_chain(uploaded_files)
    return chain

if __name__ == "__main__":
    result_chain = main()
    print(result_chain)
