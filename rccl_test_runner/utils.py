import json, io
import jsonlines


def read_json_records(json_bytes: bytes):
    """
    Parse bytes as JSON array/object or line-delimited JSON.
    Returns a list of dicts.
    """
    try:
        data = json.loads(json_bytes)
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
    except Exception:
        buf = io.BytesIO(json_bytes)
        reader = jsonlines.Reader(buf)
        return list(reader)
    return []
