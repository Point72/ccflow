import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from ccflow import FileHandler


def test_file_handler():
    with TemporaryDirectory() as tempdir:
        # Make an arbitrary path in a temporary directory
        output_file = Path(tempdir) / "a" / "random" / "sub" / "path" / "file.log"
        assert not output_file.exists(), "Output file should not exist before the test"

        # Attach handler to loggers
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        handler = FileHandler(str(output_file))
        logger.addHandler(handler)

        # Print some stuff
        logger.info("Test log message")

        # Assert everything is ok
        assert output_file.exists()
        assert "Test log message" in output_file.read_text()
