"""
DashVector Vector Database integration.
"""
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass
import asyncio

try:
    import dashvector
except ImportError:
    raise ImportError("`dashvector` not installed. Use `pip install dashvector` to install it.")

from agno.vectordb.base import VectorDb
from agno.vectordb.search import SearchType
from agno.knowledge.document import Document
from agno.knowledge.embedder.base import Embedder


@dataclass
class DashVectorConfig:
    """Configuration for DashVector client."""
    api_key: str
    endpoint: str
    protocol: Any = None  # dashvector.DashVectorProtocol
    timeout: float = 10.0


class DashVectorDb(VectorDb):
    """
    DashVector vector database integration for agno.
    
    Args:
        collection: Name of the collection
        embedder: Embedder instance for generating embeddings
        config: DashVectorConfig instance with API credentials
        dimension: Vector dimension (required if creating new collection)
        metric: Distance metric (cosine, dotproduct, euclidean)
    """

    def __init__(
        self,
        collection: str,
        embedder: Embedder,
        config: DashVectorConfig,
        dimension: Optional[int] = None,
        metric: str = "cosine",
    ):
        self.collection_name = collection
        self.embedder = embedder
        self.config = config
        self.dimension = dimension
        self.metric = metric

        
        # Initialize DashVector client
        protocol = config.protocol or dashvector.DashVectorProtocol.GRPC
        self.client = dashvector.Client(
            api_key=config.api_key,
            endpoint=config.endpoint,
            protocol=protocol,
            timeout=config.timeout,
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        collection = self.client.get(name=self.collection_name)
        if collection is None:
            if self.dimension is None:
                raise ValueError("Dimension is required to create a new collection")
            rsp = self.client.create(
                name=self.collection_name,
                dimension=self.dimension,
                metric=self.metric,
            )
            if not rsp:
                raise RuntimeError(f"Failed to create collection: {self.collection_name}")
            collection = self.client.get(name=self.collection_name)
        return collection

    def _document_to_dashvector_doc(self, document: Document) -> dashvector.Doc:
        """Convert agno Document to DashVector Doc."""
        # Generate embedding if not present
        if document.embedding is None:
            print(f"[DashVector] Generating embedding for document: {document.id or 'no-id'}")
            document.embedding = self.embedder.get_embedding(document.content)
            print(f"[DashVector] Embedding generated, dimension: {len(document.embedding)}")
        
        # Prepare fields - include content for retrieval
        fields = {"content": document.content}
        if document.name:
            fields["name"] = document.name
        if document.meta_data:
            fields.update({k: v for k, v in document.meta_data.items() if v is not None})
        
        doc_id = document.id or str(hash(document.content))
        
        return dashvector.Doc(
            id=doc_id,
            vector=document.embedding,
            fields=fields,
        )

    def get_supported_search_types(self) -> Set[SearchType]:
        """Return supported search types."""
        return {SearchType.vector}

    def create(self) -> None:
        """Create the collection if it doesn't exist."""
        if not self.exists():
            if self.dimension is None:
                raise ValueError("Dimension is required to create a new collection")
            print(f"Creating collection: {self.collection_name}, dimension={self.dimension}, metric={self.metric}")
            try:
                rsp = self.client.create(
                    name=self.collection_name,
                    dimension=self.dimension,
                    metric=self.metric,
                )
                print(f"Create response: {rsp}")
                if not rsp:
                    raise RuntimeError(f"Failed to create collection: {self.collection_name}")
                if hasattr(rsp, 'code') and rsp.code != 0:
                    raise RuntimeError(f"Failed to create collection: {self.collection_name}, code={rsp.code}, message={getattr(rsp, 'message', 'unknown')}")
                self.collection = self.client.get(name=self.collection_name)
                print(f"Collection created successfully: {self.collection}")
            except Exception as e:
                print(f"Error creating collection: {e}")
                raise

    async def async_create(self) -> None:
        """Async version of create."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.create)

    def drop(self) -> None:
        """Drop the collection."""
        self.client.delete(name=self.collection_name)

    async def async_drop(self) -> None:
        """Async version of drop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.drop)

    def exists(self) -> bool:
        """Check if the collection exists."""
        try:
            collections = self.client.list()
            return self.collection_name in collections
        except Exception:
            return False

    async def async_exists(self) -> bool:
        """Async version of exists."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.exists)

    def insert(self, documents: List[Document]) -> None:
        """Insert documents into the collection."""
        print(f"[DashVector] Insert called with {len(documents)} documents")
        if not documents:
            print("[DashVector] No documents to insert")
            return
        print(f"[DashVector] Converting {len(documents)} documents to DashVector format...")
        dash_docs = [self._document_to_dashvector_doc(doc) for doc in documents]
        print(f"[DashVector] Inserting {len(dash_docs)} docs into collection...")
        rsp = self.collection.insert(dash_docs)
        print(f"[DashVector] Insert response: success={bool(rsp)}")
        if not rsp:
            raise RuntimeError(f"Failed to insert documents: {rsp}")
        print("[DashVector] Insert completed successfully")

    async def async_insert(self, documents: List[Document]) -> None:
        """Async version of insert."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.insert, documents)

    def upsert(self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        """Upsert documents into the collection."""
        print(f"[DashVector] Upsert called with content_hash={content_hash}, {len(documents)} documents")
        if not documents:
            print("[DashVector] No documents to upsert")
            return
        print(f"[DashVector] Converting {len(documents)} documents to DashVector format...")
        dash_docs = [self._document_to_dashvector_doc(doc) for doc in documents]
        print(f"[DashVector] Upserting {len(dash_docs)} docs into collection...")
        rsp = self.collection.upsert(dash_docs)
        print(f"[DashVector] Upsert response: success={bool(rsp)}")
        if not rsp:
            raise RuntimeError(f"Failed to upsert documents: {rsp}")
        print("[DashVector] Upsert completed successfully")

    async def async_upsert(self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        """Async version of upsert."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.upsert, content_hash, documents, filters)

    def search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        """Search for similar documents."""
        # Generate query embedding
        query_embedding = self.embedder.get_embedding(query)
        
        # Query the collection
        rsp = self.collection.query(
            vector=query_embedding,
            topk=limit,
            filter=filters,
        )
        
        if not rsp:
            return []
        
        # Convert DashVector docs to agno Documents
        documents = []
        for doc in rsp.output:
            meta_data = dict(doc.fields) if doc.fields else {}
            content = meta_data.pop("content", "")
            documents.append(
                Document(
                    id=doc.id,
                    content=content,
                    meta_data=meta_data,
                    embedding=doc.vector,
                )
            )
        return documents

    async def async_search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        """Async version of search."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.search(query, limit, filters, **kwargs))

    def delete(self, document_ids: Optional[List[str]] = None, filters: Optional[Dict[str, Any]] = None) -> None:
        """Delete documents from the collection."""
        if document_ids:
            for doc_id in document_ids:
                self.collection.delete(doc_id)

    def delete_by_id(self, document_id: str) -> None:
        """Delete a document by ID."""
        self.collection.delete(document_id)

    def delete_by_content_id(self, content_id: str) -> None:
        """Delete documents by content ID."""
        self.collection.delete(content_id)

    def delete_by_name(self, name: str) -> None:
        """Delete documents by name."""
        pass

    def delete_by_metadata(self, metadata_filter: Dict[str, Any]) -> None:
        """Delete documents by metadata filter."""
        pass

    def id_exists(self, document_id: str) -> bool:
        """Check if a document ID exists."""
        try:
            rsp = self.collection.fetch(document_id)
            return rsp and rsp.output is not None
        except Exception:
            return False

    async def async_name_exists(self, name: str) -> bool:
        """Async version of name_exists."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.name_exists, name)

    def name_exists(self, name: str) -> bool:
        """Check if a document name exists."""
        return False

    def content_hash_exists(self, content_hash: str) -> bool:
        """Check if a content hash exists."""
        # Store content_hash in metadata and check
        try:
            rsp = self.collection.query(
                vector=[0.0] * self.dimension,
                topk=1,
                filter={"content_hash": content_hash}
            )
            return rsp and rsp.output and len(rsp.output) > 0
        except Exception:
            return False

    def optimize(self) -> None:
        """Optimize the collection - not applicable for DashVector."""
        pass

    def clear(self) -> bool:
        """Clear all documents from the collection."""
        try:
            stats = self.collection.stats()
            if stats and stats.output and stats.output.total_doc_count:
                self.client.delete(name=self.collection_name)
                self.collection = self._get_or_create_collection()
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False

    def get_count(self) -> int:
        """Get the number of documents in the collection."""
        stats = self.collection.stats()
        if stats and stats.output and stats.output.total_doc_count:
            return int(stats.output.total_doc_count)
        return 0

    def update_metadata(self, document_id: str, metadata: Dict[str, Any]) -> None:
        """Update document metadata."""
        pass

    def upsert_available(self) -> bool:
        """Check if upsert is available."""
        return True
