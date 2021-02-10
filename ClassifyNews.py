from pydantic import BaseModel

class ClassifyNews(BaseModel):
    newsSummary : str
