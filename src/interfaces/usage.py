from pydantic import BaseModel


class UserContext(BaseModel):
    key: str
    model_name: str
    endpoint: str


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cached_tokens: int


class UsageFullData(UserContext, Usage):
    pass
