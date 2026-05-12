from datetime import datetime, timezone


def timestamp_ms_to_string(ts_ms: int) -> str:
    try:
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    except (ValueError, OSError, OverflowError):
        return f"ts={ts_ms}"
