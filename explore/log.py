import logging
import click

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
        
        logger = structlog.get_logger()
        structlog_bench(logger, num_iterations)

    else:
        import json
        from datetime import datetime
        
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "context": {k: record.__dict__.get(k) for k in record.__dict__ if k not in {'message', 'args', 'levelname', 'levelno', 'pathname', 'filename', 'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated', 'thread', 'threadName', 'processName', 'process'}}
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
        
        logger = configure_logger()
        standard_bench(logger, num_iterations)



if __name__ == "__main__":
    run_bench()
