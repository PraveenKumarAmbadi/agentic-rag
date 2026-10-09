"""Elasticsearch client factory — single source of truth for ES connections."""

from elasticsearch import Elasticsearch

from configs.loader import load_config


def get_es_client(cfg: dict | None = None) -> Elasticsearch:
    """Return a connected Elasticsearch client.

    Args:
        cfg: Validated app config; loaded from config.yaml when omitted.
            ES_HOST / ES_PORT overrides are applied by the config loader.

    Raises:
        ConnectionError: if Elasticsearch is not reachable.
    """
    cfg = cfg or load_config()
    host = cfg["elasticsearch"]["host"]
    port = cfg["elasticsearch"]["port"]
    client = Elasticsearch(f"http://{host}:{port}")

    if not client.ping():
        raise ConnectionError(
            f"Elasticsearch at http://{host}:{port} is not reachable. "
            "Start Elasticsearch, or point ES_HOST / ES_PORT at it."
        )
    return client


if __name__ == "__main__":
    client = get_es_client()
    info = client.info()
    print(f"Connected to Elasticsearch {info['version']['number']}")
