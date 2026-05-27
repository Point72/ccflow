"""Unit tests for ccflow.ui.cli module."""

from ccflow.ui.cli import _get_ui_args_parser


class TestGetUIArgsParser:
    """Tests for _get_ui_args_parser function.

    Note: Default values for hydra config args and panel server args are tested
    in ccflow/tests/utils/test_hydra.py. These tests focus on viewer-specific
    arguments and verifying the parser composition works correctly.
    """

    def test_parser_composition(self):
        """Test parser includes args from both helper functions."""
        parser = _get_ui_args_parser()
        args = parser.parse_args([])

        # From add_hydra_config_args
        assert hasattr(args, "overrides")
        assert hasattr(args, "config_path")
        assert hasattr(args, "config_name")

        # From add_panel_server_args
        assert hasattr(args, "address")
        assert hasattr(args, "port")
        assert hasattr(args, "show")

        # Viewer-specific
        assert hasattr(args, "browser_width")
        assert hasattr(args, "title")

    def test_viewer_layout_defaults(self):
        """Test default values for viewer-specific arguments."""
        parser = _get_ui_args_parser()
        args = parser.parse_args([])

        assert args.browser_width == 400
        assert args.title == "ccflow Model Registry"

    def test_viewer_layout_custom_values(self):
        """Test setting custom values for viewer layout arguments."""
        parser = _get_ui_args_parser()
        args = parser.parse_args(
            [
                "--browser-width",
                "500",
                "--title",
                "My Registry",
            ]
        )

        assert args.browser_width == 500
        assert args.title == "My Registry"

    def test_overrides_positional(self):
        """Test overrides are captured as positional arguments."""
        parser = _get_ui_args_parser()
        args = parser.parse_args(["key1=value1", "key2=value2"])

        assert args.overrides == ["key1=value1", "key2=value2"]
