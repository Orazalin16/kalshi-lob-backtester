"""Request signing for the Kalshi API.

Every authenticated request carries three headers. The signature covers
``timestamp_ms + METHOD + path`` where the path includes ``/trade-api/v2`` and
excludes the query string. Kalshi verifies with whichever algorithm matches the
registered public key: Ed25519 signs the text directly, RSA uses PSS with
SHA-256 and a salt as long as the digest.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path


class Signer:
    def __init__(self, key_id: str, private_key_pem: bytes) -> None:
        from cryptography.hazmat.primitives import serialization

        self.key_id = key_id
        self._key = serialization.load_pem_private_key(private_key_pem, password=None)

    @classmethod
    def from_file(cls, key_id: str, path: str | Path) -> "Signer":
        return cls(key_id, Path(path).read_bytes())

    def sign(self, text: str) -> str:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        message = text.encode()
        if isinstance(self._key, Ed25519PrivateKey):
            sig = self._key.sign(message)
        else:
            pss = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH)
            sig = self._key.sign(message, pss, hashes.SHA256())
        return base64.b64encode(sig).decode()

    def headers(self, method: str, path: str, now_ms: int | None = None) -> dict[str, str]:
        ts = str(int(time.time() * 1000) if now_ms is None else now_ms)
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self.sign(ts + method.upper() + path.split("?")[0]),
        }
