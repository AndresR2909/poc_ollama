import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class VectorStore:
    def __init__(
        self,
        path: str,
        vector_store: Chroma = None,
        emmbedded_model: OllamaEmbeddings = None,
        llm=None,
    ):
        if llm is None:
            raise ValueError("Por favor, proporciona un modelo de lenguaje.")
        if vector_store is None:
            raise ValueError("Por favor, proporciona un vector store.")
        if emmbedded_model is None:
            raise ValueError("Por favor, proporciona un modelo emmbedded.")
        self.llm = llm
        self.emmbedded_model = emmbedded_model
        self.vector_store = vector_store
        self.path = path
        self.vdb = None

    def load_documents(self):
        """
        Carga los documentos en el vector store.
        """
        documents = []
        pdf_filenames = os.listdir(self.path)
        for pdf_filename in pdf_filenames:
            pdf_path = self.path + "/" + pdf_filename
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        return documents

    def split_documents(
        self, documents, chunk_size=2000, chunk_overlap=200, recursive=False
    ):
        """
        Divide los documentos en oraciones.
        """
        separators = [
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ]
        if recursive:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=separators,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        return text_splitter.split_documents(documents)

    def create_vector_store(self):
        """
        Almacena los vectores en el vector store.
        """
        # Cargar y combinar documentos
        documents = self.load_documents()
        # Dividir documentos
        splitted_documents = self.split_documents(documents)

        # Crear vectorstore
        self.vdb = self.vector_store.from_documents(
            documents=splitted_documents, embedding=self.emmbedded_model
        )

        # Configurar retriever
        return self.vdb

    def get_retriever(self):
        if self.vdb is None:
            self.create_vector_store()
        return self.vdb.as_retriever()
