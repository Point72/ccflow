"""Unit tests for ccflow.ui.cli module."""

from ccflow.ui.cli import _get_ui_args_parser


class TestGetUIArgsParser:
    """Tests for _get_ui_args_parser function."""

    def test_parser_has_config_args(self):
        """Test parser includes config arguments."""
        parser = _get_ui_args_parser()

        # Parse with config args
        args = parser.parse_args(
            [
                "--config-path",
                "/path/to/config",
                "--config-name",
                "base",
            ]
        )

        assert args.config_path == "/path/to/config"
        assert args.config_name == "base"

    def test_parser_defaults(self):
        """Test parser default values."""
        parser = _get_ui_args_parser()
        args = parser.parse_args(["-cp", ".", "-cn", "test"])

        assert args.address == "127.0.0.1"
        assert args.port == 8080
        assert args.browser_width == 400
        assert args.browser_height == 700
        assert args.viewer_width is None
        assert args.show is False

    def test_parser_overrides(self):
        """Test parser accepts override arguments."""
        parser = _get_ui_args_parser()
        args = parser.parse_args(
            [
                "-cp",
                ".",
                "-cn",
                "test",
                "key1=value1",
                "key2=value2",
            ]
        )

        assert args.overrides == ["key1=value1", "key2=value2"]

    def test_parser_ui_args(self):
        """Test parser UI server arguments."""
        parser = _get_ui_args_parser()
        args = parser.parse_args(
            [
                "-cp",
                ".",
                "-cn",
                "test",
                "--address",
                "0.0.0.0",
                "--port",
                "9000",
                "--show",
            ]
        )

        assert args.address == "0.0.0.0"
        assert args.port == 9000
        assert args.show is True

    def test_parser_viewer_layout_args(self):
        """Test parser viewer layout arguments."""
        parser = _get_ui_args_parser()
        args = parser.parse_args(
            [
                "-cp",
                ".",
                "-cn",
                "test",
                "--browser-width",
                "500",
                "--browser-height",
                "800",
                "--viewer-width",
                "600",
            ]
        )

        assert args.browser_width == 500
        assert args.browser_height == 800
        assert args.viewer_width == 600

    def test_parser_websocket_origin(self):
        """Test parser websocket origin argument."""
        parser = _get_ui_args_parser()
        args = parser.parse_args(
            [
                "-cp",
                ".",
                "-cn",
                "test",
                "--allow-websocket-origin",
                "localhost:8080",
                "example.com",
            ]
        )

        assert args.allow_websocket_origin == ["localhost:8080", "example.com"]

    def test_parser_config_dir_args(self):
        """Test parser config directory arguments."""
        parser = _get_ui_args_parser()
        args = parser.parse_args(
            [
                "-cp",
                "/root/config",
                "-cn",
                "base",
                "-cd",
                "/extra/config",
                "-cdcn",
                "override",
                "--basepath",
                "/search/from/here",
            ]
        )

        assert args.config_path == "/root/config"
        assert args.config_name == "base"
        assert args.config_dir == "/extra/config"
        assert args.config_dir_config_name == "override"
        assert args.basepath == "/search/from/here"
