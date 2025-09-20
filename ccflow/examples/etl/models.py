import sqlite3
from csv import DictReader, DictWriter
from io import StringIO
from typing import Optional

from bs4 import BeautifulSoup
from httpx import Client
from pydantic import Field

from ccflow import CallableModel, ContextBase, Flow, GenericResult, NullContext

__all__ = ("RestModel", "LinksModel", "DBModel", "SiteContext")


class SiteContext(ContextBase):
    """An example of a context object, passed into and between callable models from the command line."""

    site: str = Field(default="https://en.wikipedia.org/wiki/Main_Page")


class RestModel(CallableModel):
    """Example callable model that fetches a URL and returns the HTML content."""

    @Flow.call
    def __call__(self, context: Optional[SiteContext] = None) -> GenericResult[str]:
        context = context or SiteContext()
        resp = Client().get(context.site, follow_redirects=True)
        resp.raise_for_status()

        return GenericResult[str](value=resp.text)


class LinksModel(CallableModel):
    """Example callable model that transforms HTML content into CSV of links."""

    file: str

    @Flow.call
    def __call__(self, context: Optional[NullContext] = None) -> GenericResult[str]:
        context = context or NullContext()

        with open(self.file, "r") as f:
            html = f.read()

        # Use beautifulsoup to convert links into csv of name, url
        soup = BeautifulSoup(html, "html.parser")
        links = [{"name": a.text, "url": href} for a in soup.find_all("a", href=True) if (href := a["href"]).startswith("http")]

        io = StringIO()
        writer = DictWriter(io, fieldnames=["name", "url"])
        writer.writeheader()
        writer.writerows(links)
        output = io.getvalue()
        return GenericResult[str](value=output)


class DBModel(CallableModel):
    """Example callable model that loads CSV data into a SQLite database."""

    file: str
    db_file: str = Field(default="etl.db")
    table: str = Field(default="links")

    @Flow.call
    def __call__(self, context: Optional[NullContext] = None) -> GenericResult[str]:
        context = context or NullContext()

        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {self.table} (name TEXT, url TEXT)")
        with open(self.file, "r") as f:
            reader = DictReader(f)
            for row in reader:
                cursor.execute(f"INSERT INTO {self.table} (name, url) VALUES (?, ?)", (row["name"], row["url"]))
        conn.commit()
        return GenericResult[str](value="Data loaded into database")
