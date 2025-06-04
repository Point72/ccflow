from logging import FileHandler as BaseFileHandler, StreamHandler
from pathlib import Path

__all__ = ("StreamHandler", "FileHandler")


class FileHandler(BaseFileHandler):
    def __init__(self, filename, mode="a", encoding=None, delay=False):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(filename, mode, encoding, delay)
