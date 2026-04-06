import sys

import ddtrace
from ddtrace import tracer
from loguru import logger as loguru_logger

CUSTOM_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "<level>{level: <8}</level> | "
    "[{name}:{line}] - "
    "<level>{extra}</level> - "
    "<level>{message}</level>"
)


class Logger:
    _logger = loguru_logger

    @classmethod
    def set_extra(cls, **kwargs):
        cls._logger.remove()
        cls._logger.add(sys.stdout, format=CUSTOM_FORMAT, serialize=True)
        span = tracer.current_span()
        trace_id, span_id = (str((1 << 64) - 1 & span.trace_id), span.span_id) if span else (None, None)

        # add ids to structlog event dictionary
        kwargs["dd.trace_id"] = str(trace_id or 0)
        kwargs["dd.span_id"] = str(span_id or 0)

        # add the env, service, and version configured for the tracer
        kwargs["dd.env"] = ddtrace.config.env or ""
        kwargs["dd.service"] = ddtrace.config.service or ""
        kwargs["dd.version"] = ddtrace.config.version or ""
        cls._logger = loguru_logger.bind(**kwargs)

    @classmethod
    def add_extra(cls, **kwargs):
        cls._logger = cls._logger.bind(**kwargs)

    def __getattr__(self, item):
        return getattr(self._logger, item)


logger: loguru_logger = Logger()
