import logging
import click

import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "level": record.levelname.lower(),
            "event": record.getMessage(),
            "i": record.__dict__.get("i")
        }
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)

def configure_logger():
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

standard_logger = configure_logger()

import orjson
import structlog
structlog.configure(
    cache_logger_on_first_use=True,
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.JSONRenderer(serializer=orjson.dumps),
    ],
    logger_factory=structlog.BytesLoggerFactory(),
)

structlog_logger = structlog.get_logger()

def structlog_bench(logger, num_iterations):
    log = logger.bind()
    for i in range(num_iterations):
       log.info("iterated", i=i)


def standard_bench(logger, num_iterations):
    for i in range(num_iterations):
        logger.info("iterated", extra={"i": i})

@click.command()
@click.option('-l', '--logger-type', default="structlog", help='Type of logger to benchmark')
@click.option('-n', '--num_iterations', default=100_000, help='Number of iterations.')
def run_bench(logger_type, num_iterations):
    if logger_type == "structlog":
        structlog_bench(structlog_logger, num_iterations)

    else:
        standard_bench(standard_logger, num_iterations)



if __name__ == "__main__":
    run_bench()
