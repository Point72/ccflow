from tempfile import NamedTemporaryFile
from unittest.mock import patch

from ccflow.examples.etl.__main__ import main
from ccflow.examples.etl.explain import explain
from ccflow.examples.etl.models import DBModel, LinksModel, RestModel, SiteContext


class TestEtl:
    def test_rest_model(self):
        rest = RestModel()
        context = SiteContext(site="https://news.ycombinator.com")
        result = rest(context)
        assert result.value is not None
        assert "Hacker News" in result.value

    def test_links_model(self):
        with NamedTemporaryFile(suffix=".html") as file:
            file.write(b"""
            <html>
                <body>
                    <a href="https://example.com/page1">Page 1</a>
                    <a href="https://example.com/page2">Page 2</a>
                </body>
            </html>
            """)
            file.flush()
            links = LinksModel(file=file.name)
            result = links()
            assert result.value is not None
            assert "name,url" in result.value  # Check for CSV header

    def test_db_model(self):
        with NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as file:
            file.write("name,url\nPage 1,https://example.com/page1\nPage 2,https://example.com/page2\n")
            file.flush()
            db = DBModel(file=file.name, db_file=":memory:", table="links")
            result = db()
            assert result.value == "Data loaded into database"

    def test_cli(self):
        with patch("ccflow.examples.etl.__main__.cfg_run") as mock_cfg_run:
            with patch("sys.argv", ["etl", "+callable=extract", "+context=[]"]):
                main()
                mock_cfg_run.assert_called_once()

    def test_explain(self):
        with patch("ccflow.examples.etl.explain.cfg_explain_cli") as mock_cfg_explain:
            explain()
            mock_cfg_explain.assert_called_once()
