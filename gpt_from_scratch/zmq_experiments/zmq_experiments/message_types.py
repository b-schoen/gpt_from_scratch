import pydantic


class Message(pydantic.BaseModel):
    content: str
