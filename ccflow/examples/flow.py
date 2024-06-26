from ccflow import BaseModel


class Source1FromDb(BaseModel):
    conn_str: str
    query: str


class Source1FromParquet(BaseModel): ...


class Source2FromDb(BaseModel):
    conn_str: str
    query: str


class Source2FromParquet(BaseModel): ...


class Other1(BaseModel): ...


class Other2(BaseModel): ...
