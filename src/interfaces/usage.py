from enum import Enum

from pydantic import BaseModel


class InferenceCallType(str, Enum):
    text = "text"
    image = "image"


class UserContext(BaseModel):
    key: str
    model_name: str
    endpoint: str


class TextUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    type: InferenceCallType | None = None  # Optional for backward compatibility


class ImageUsage(BaseModel):
    image_count: int
    type: InferenceCallType = InferenceCallType.image


class TextUsageFullData(UserContext, TextUsage):
    pass


class ImageUsageFullData(UserContext, ImageUsage):
    pass


# Deprecated: Keep for backward compatibility
class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cached_tokens: int


class UsageFullData(UserContext, Usage):
    pass
