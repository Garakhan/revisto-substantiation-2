"""Elasticsearch operations"""

from typing import Dict, List, Any, Optional, Iterator
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import RequestError

from ..config import ESConfig, EmbedConfig
from ..utils.logging import get_logger
from .mappings import get_index_mapping, get_index_settings

logger = get_logger(__name__)


class ESClient:
    """Elasticsearch client wrapper"""
    
    def __init__(self, config: ESConfig):
        self.config = config
        self._client = None
    
    @property
    def client(self) -> Elasticsearch:
        """Get or create ES client"""
        if self._client is None:
            if self.config.use_api_key:
                self._client = Elasticsearch(
                    self.config.url,
                    api_key=self.config.api_key,
                    verify_certs=self.config.verify_certs,
                )
            else:
                self._client = Elasticsearch(
                    self.config.url,
                    basic_auth=(self.config.user, self.config.password),
                    verify_certs=self.config.verify_certs,
                )
            
            # Test connection
            if not self._client.ping():
                raise ConnectionError(f"Failed to connect to Elasticsearch at {self.config.url}")
            
            logger.info(f"Connected to Elasticsearch at {self.config.url}")
        
        return self._client
    
    def create_index(
        self,
        index_name: str,
        mapping: Dict[str, Any],
        settings: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create an index with mapping and settings"""
        if settings is None:
            settings = get_index_settings(self.config)
        
        body = {
            "settings": settings,
            "mappings": mapping
        }
        
        try:
            result = self.client.indices.create(index=index_name, body=body)
            logger.info(f"Created index: {index_name}")
            return True
        except RequestError as e:
            if "resource_already_exists_exception" in str(e):
                logger.warning(f"Index {index_name} already exists")
                return False
            raise
    
    def create_alias(self, index_name: str, alias_name: str) -> bool:
        """Create or update an alias"""
        try:
            # Remove alias from other indices if it exists
            if self.client.indices.exists_alias(name=alias_name):
                current = self.client.indices.get_alias(name=alias_name)
                for idx in current:
                    self.client.indices.delete_alias(index=idx, name=alias_name)
            
            # Add alias to new index
            self.client.indices.put_alias(index=index_name, name=alias_name)
            logger.info(f"Created alias {alias_name} -> {index_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating alias: {e}")
            return False
    
    def bulk_index(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        batch_size: int = 500
    ) -> Dict[str, int]:
        """Bulk index documents"""
        success_count = 0
        error_count = 0
        
        def generate_actions():
            for doc in documents:
                yield {
                    "_index": index_name,
                    "_source": doc
                }
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = list(generate_actions())[i:i + batch_size]
            
            try:
                stats = helpers.bulk(
                    self.client,
                    batch,
                    stats_only=True,
                    raise_on_error=False
                )
                success_count += stats[0]
                error_count += stats[1]
                
                if i % (batch_size * 10) == 0:
                    logger.info(f"Indexed {i + len(batch)} / {len(documents)} documents")
                    
            except Exception as e:
                logger.error(f"Error in bulk indexing: {e}")
                error_count += len(batch)
        
        logger.info(f"Bulk indexing complete: {success_count} successful, {error_count} errors")
        
        return {"success": success_count, "errors": error_count}
    
    def search(
        self,
        index_name: str,
        query: Dict[str, Any],
        size: int = 10,
        source_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute a search query"""
        body = {"query": query, "size": size}
        
        if source_fields:
            body["_source"] = source_fields
        
        return self.client.search(index=index_name, body=body)
    
    def scroll_search(
        self,
        index_name: str,
        query: Dict[str, Any],
        size: int = 1000,
        scroll_time: str = "5m"
    ) -> Iterator[Dict[str, Any]]:
        """Scroll through all results"""
        # Initial search
        response = self.client.search(
            index=index_name,
            body={"query": query, "size": size},
            scroll=scroll_time
        )
        
        # Yield initial results
        for hit in response["hits"]["hits"]:
            yield hit["_source"]
        
        # Continue scrolling
        scroll_id = response["_scroll_id"]
        
        while True:
            response = self.client.scroll(
                scroll_id=scroll_id,
                scroll=scroll_time
            )
            
            if not response["hits"]["hits"]:
                break
            
            for hit in response["hits"]["hits"]:
                yield hit["_source"]
        
        # Clear scroll
        self.client.clear_scroll(scroll_id=scroll_id)
    
    def count(self, index_name: str, query: Optional[Dict[str, Any]] = None) -> int:
        """Count documents matching query"""
        if query is None:
            query = {"match_all": {}}
        
        result = self.client.count(index=index_name, body={"query": query})
        return result["count"]
    
    def get_ref_metadata(
        self,
        index_name: str,
        ref_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch ref_metadata documents for a list of ref_ids.

        Returns:
            Dict mapping ref_id -> doc_metadata dict
        """
        if not ref_ids:
            return {}

        query = {
            "bool": {
                "filter": [
                    {"term": {"doc_type": "ref_metadata"}},
                    {"terms": {"ref_id": list(set(ref_ids))}}
                ]
            }
        }

        try:
            response = self.client.search(
                index=index_name,
                body={"query": query, "size": len(set(ref_ids))},
            )
            result = {}
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                result[source["ref_id"]] = source.get("doc_metadata", {})
            return result
        except Exception as e:
            logger.error(f"Error fetching ref metadata: {e}")
            return {}

    def delete_index(self, index_name: str) -> bool:
        """Delete an index"""
        try:
            self.client.indices.delete(index=index_name)
            logger.info(f"Deleted index: {index_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            return False


# Convenience functions
def create_es_client(config: Optional[ESConfig] = None) -> ESClient:
    """Create an ES client"""
    if config is None:
        config = ESConfig()
    return ESClient(config)


def create_index(
    client: ESClient,
    index_name: str,
    embed_config: EmbedConfig
) -> bool:
    """Create index with proper mapping"""
    mapping = get_index_mapping(embed_config)
    settings = get_index_settings(client.config)
    return client.create_index(index_name, mapping, settings)


def bulk_index(
    client: ESClient,
    index_name: str,
    documents: List[Dict[str, Any]],
    batch_size: int = 500
) -> Dict[str, int]:
    """Bulk index documents"""
    return client.bulk_index(index_name, documents, batch_size)


def search_documents(
    client: ESClient,
    index_name: str,
    query: Dict[str, Any],
    size: int = 10
) -> List[Dict[str, Any]]:
    """Search and return documents"""
    results = client.search(index_name, query, size)
    return [hit["_source"] for hit in results["hits"]["hits"]]