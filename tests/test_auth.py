import base64

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa  # noqa: E402

from klob.kalshi.auth import Signer  # noqa: E402


def pem(key):
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())


def test_headers_sign_timestamp_method_and_path_without_query():
    key = ed25519.Ed25519PrivateKey.generate()
    h = Signer("key-id", pem(key)).headers("get", "/trade-api/v2/portfolio/orders?limit=5", now_ms=1700000000000)
    assert h["KALSHI-ACCESS-KEY"] == "key-id" and h["KALSHI-ACCESS-TIMESTAMP"] == "1700000000000"
    signed = b"1700000000000GET/trade-api/v2/portfolio/orders"
    key.public_key().verify(base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]), signed)  # raises if wrong


def test_rsa_keys_use_pss_with_digest_length_salt():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    h = Signer("k", pem(key)).headers("GET", "/trade-api/ws/v2", now_ms=42)
    key.public_key().verify(
        base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]),
        b"42GET/trade-api/ws/v2",
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
