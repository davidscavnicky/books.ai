import logging

import booksai._version

__version__ = booksai._version.__version__


logger = logging.getLogger(__name__)
logger.info("Imported booksai version: %s", __version__)
