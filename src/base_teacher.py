from langchain import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class BaseTeacher:
    def __init__(
        self, topic_prompt_template, chat_prompt_template, llm=None, retriever=None
    ):
        if llm is None:
            # Inicializar el modelo de lenguaje con los parámetros deseados
            raise ValueError("Por favor, proporciona un modelo de lenguaje.")
        self.llm = llm
        if retriever is None:
            # Inicializar el modelo de lenguaje con los parámetros deseados
            raise ValueError("Por favor, proporciona un retriever.")
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
        self.retriever = retriever
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

    @staticmethod
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    def setup_knowledge_base(self):
        """
        Configura el RAG utilizando una lista de rutas a archivos PDF.
        """
        # Configurar cadena de QA con memoria
        template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        self.qa_chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def chat_with_knowledge_base(self, user_input):
        """
        Chatea sobre un tema basado en los documentos cargados (RAG).
        """
        if self.qa_chain is None:
            raise ValueError(
                "Base de conocimiento no configurada. Por favor, llama a setup_knowledge_base primero."
            )

        return self.qa_chain.invoke(user_input)
