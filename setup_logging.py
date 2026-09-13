import logging

def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging: your code talks, libraries stay quiet."""
    # 1. Root handler — controls the output format
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 2. Quiet down chatty third-party loggers
    noisy_loggers = [
        "elastic_transport",   # ES HTTP request logs
        "elasticsearch",       # Older ES client logs
        "urllib3",             # HTTP connection pool logs
        "requests",            # If you use requests elsewhere
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    return logger

if __name__ == "__main__":
    setup_logging()
    logging.info("Logging is configured.")
    