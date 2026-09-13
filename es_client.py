"""Elasticsearch client factory — single source of truth for ES connections."""

from elasticsearch import Elasticsearch
from configs.loader import load_config
import os

def get_es_client() -> Elasticsearch:
    """Return a connected Elasticsearch client using values from config.yaml.

    Returns:
        Elasticsearch client instance.

    Raises:
        ConnectionError: if Elasticsearch is not reachable.
    """

    config = load_config()
        
    host = os.getenv("ES_HOST", config["elasticsearch"]["host"])
    port = os.getenv("ES_PORT", str(config["elasticsearch"]["port"]))
    es_URL  = f"http://{host}:{port}"
    client  = Elasticsearch(es_URL)
    
    if not client.ping():
        raise ConnectionError(
            f"Elasticsearch at http://{host}:{port} is not reachable. "
            "Is the container running? Try: docker compose up -d"
        )
    return client 

if __name__ == "__main__":
    client = get_es_client()
    info = client.info()
    print(f"Connected to Elasticsearch {info['version']['number']}")
