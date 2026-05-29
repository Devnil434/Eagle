from __future__ import annotations

import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def retry_on_failure(max_retries: int = 3, delay: int = 2):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            retry_count = 0

            while retry_count <= max_retries:

                try:
                    return func(*args, **kwargs)

                except Exception as e:

                    retry_count += 1

                    logger.warning(
                        "Retry %d/%d for %s due to %s",
                        retry_count,
                        max_retries,
                        func.__name__,
                        str(e),
                    )

                    if retry_count > max_retries:
                        raise

                    time.sleep(delay)

        return wrapper

    return decorator