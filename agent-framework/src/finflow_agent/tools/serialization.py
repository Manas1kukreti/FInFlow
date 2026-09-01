from pydantic import BaseModel
import json

def safe_model_dump(model: BaseModel, **kwargs) -> dict:
    """
    Safely serialize a Pydantic model to a dictionary, supporting both v1 and v2 APIs.
    """
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)
    elif hasattr(model, "dict"):
        return model.dict(**kwargs)
    else:
        raise ValueError(f"Object {type(model)} does not appear to be a Pydantic model.")

def safe_model_dump_json(model: BaseModel, **kwargs) -> str:
    """
    Safely serialize a Pydantic model to a JSON string, supporting both v1 and v2 APIs.
    """
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(**kwargs)
    elif hasattr(model, "json"):
        return model.json(**kwargs)
    else:
        raise ValueError(f"Object {type(model)} does not appear to be a Pydantic model.")
