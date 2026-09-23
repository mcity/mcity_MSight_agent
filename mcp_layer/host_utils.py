import logging

import requests

# Cached for the process lifetime -- whether this host is on EC2, and what its
# public IP is, can't change mid-run. Without caching, a high-frequency caller
# (e.g. the dashboard's status polling, every 4s) would pay IMDS's ~2s timeout
# on every single call -- confirmed happening live before this fix.
_imds_token: str | None = None
_imds_checked = False
_resolved_host: str | None = None


def get_imds_token() -> str | None:
    global _imds_token, _imds_checked
    if _imds_checked:
        return _imds_token
    _imds_checked = True
    token_url = "http://169.254.169.254/latest/api/token"
    headers = {"X-aws-ec2-metadata-token-ttl-seconds": "21600"}
    try:
        response = requests.put(token_url, headers=headers, timeout=2)
        response.raise_for_status()
        _imds_token = response.text
    except Exception as e:
        logging.warning(f"Error getting IMDS token: {e}")
    return _imds_token


def get_metadata_with_token(path: str, token: str) -> str | None:
    url = f"http://169.254.169.254/latest/meta-data/{path}"
    headers = {"X-aws-ec2-metadata-token": token}
    try:
        response = requests.get(url, headers=headers, timeout=2)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logging.warning(f"Error fetching metadata for {path}: {e}")
        return None


def resolve_host() -> str:
    """Resolve the public IP via IMDSv2, falling back to localhost."""
    global _resolved_host
    if _resolved_host is not None:
        return _resolved_host
    token = get_imds_token()
    if token:
        host = get_metadata_with_token("public-ipv4", token)
        if host:
            _resolved_host = host
            return _resolved_host
    logging.warning("Could not obtain IMDSv2 token — falling back to localhost.")
    _resolved_host = "localhost"
    return _resolved_host


def is_cloud_deployment() -> bool:
    """True if running on an EC2 instance -- used to decide whether a local host filesystem path is meaningful to offer the user."""
    return get_imds_token() is not None
