import pytest

from klob.fixedpoint import price_to_ticks, qty_to_units


@pytest.mark.parametrize(
    "text, ticks",
    [("0.1200", 1200), ("0.9600", 9600), ("1.0000", 10000), ("0.0001", 1), ("0.5", 5000), ("0", 0), (".25", 2500), ("0.00550", 55)],
)
def test_prices(text, ticks):
    assert price_to_ticks(text) == ticks


@pytest.mark.parametrize("text, units", [("10.00", 1000), ("-54.00", -5400), ("1.55", 155), ("21469.40", 2146940), ("7", 700)])
def test_sizes(text, units):
    assert qty_to_units(text) == units


def test_no_float_rounding():
    # 0.07 is not representable in binary; through float this would come out as 6.
    assert qty_to_units("0.07") == 7
    assert price_to_ticks("0.2900") == 2900


@pytest.mark.parametrize("bad", ["", "abc", "1.2.3", "0.12345", "1e-3", "--1"])
def test_rejects_malformed_or_too_precise(bad):
    with pytest.raises(ValueError):
        price_to_ticks(bad)
