"""klob: order-book replay and queue-position backtesting for Kalshi markets."""

from . import reference, tape
from .fixedpoint import PRICE_MAX, PRICE_SCALE, QTY_SCALE, price_to_ticks, qty_to_units
from .tape import EVENT_DTYPE, NO, YES

try:
    from ._core import FILL_DTYPE, Backtester, Book, Order, OrderState, QueueModel, SimConfig

    HAVE_CORE = True
except ImportError:  # the C++ extension has not been built; the reference still works
    HAVE_CORE = False

__version__ = "0.1.0"

__all__ = [
    "Backtester",
    "Book",
    "EVENT_DTYPE",
    "FILL_DTYPE",
    "HAVE_CORE",
    "NO",
    "Order",
    "OrderState",
    "PRICE_MAX",
    "PRICE_SCALE",
    "QTY_SCALE",
    "QueueModel",
    "SimConfig",
    "YES",
    "price_to_ticks",
    "qty_to_units",
    "reference",
    "tape",
]
