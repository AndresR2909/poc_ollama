from langchain import PromptTemplate, LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from langchain_core.output_parsers import StrOutputParser


class BaseTeacher:
    def __init__(self, topic_prompt_template, chat_prompt_template, llm=None):
        if llm is None:
            # Inicializar el modelo de lenguaje con los parámetros deseados
            raise ValueError("Por favor, proporciona un modelo de lenguaje.")
        self.llm = llm
        self.topic_prompt = PromptTemplate(
            input_variables=["topic"],
            template=topic_prompt_template,
        )
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", chat_prompt_template),
                ("human", "{user_input}"),
            ]
        )
        # chain tema específico
        self.topic_chain = self.topic_prompt | self.llm | StrOutputParser()

        # Chain el chat abierto
        self.chat_chain = self.chat_prompt | self.llm | StrOutputParser()

        # Inicializar memoria para el chat
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        # Inicializar retriever y cadena para RAG
        self.retriever = None
        self.qa_chain = None

    def teach(self, topic):
        """
        Enseña un tema específico utilizando el prompt personalizado.
        """
        return self.topic_chain.invoke(input=topic)

    def chat(self, user_input):
        """
        Método para chatear libremente con el agente.
        """
        return self.chat_chain.invoke(user_input)

    def setup_knowledge_base(self, pdf_paths):
        """
        Configura el RAG utilizando una lista de rutas a archivos PDF.
        """
        # Cargar y combinar documentos
        documents = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load_and_split())
        # Crear embeddings y vectorstore
        embeddings = None  # OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        # Configurar retriever
        self.retriever = vectorstore.as_retriever()
        # Configurar cadena de QA con memoria
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
        )

    def chat_with_knowledge_base(self, user_input):
        """
        Chatea sobre un tema basado en los documentos cargados (RAG).
        """
        if self.qa_chain is None:
            raise ValueError(
                "Base de conocimiento no configurada. Por favor, llama a setup_knowledge_base primero."
            )
        result = self.qa_chain({"question": user_input})
        return result["answer"]
