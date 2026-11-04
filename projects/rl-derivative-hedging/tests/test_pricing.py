import numpy as np
from rl_hedging.pricing import call_price, call_delta


def test_at_expiry_intrinsic_value():
    assert call_price(110, 100, 0, 0.0, 0.2) == 10.0
    assert call_price(90, 100, 0, 0.0, 0.2) == 0.0


def test_atm_call_delta_is_between_zero_and_one():
    delta = call_delta(100, 100, 1.0, 0.0, 0.2)
    assert 0.0 < delta < 1.0


def test_deep_itm_delta_close_to_one():
    assert call_delta(300, 100, 1.0, 0.0, 0.2) > 0.99
