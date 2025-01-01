from propy.propeller import Propeller
from pytest import raises

def test_instantiation():
    """Check whether instantiation of an abstract Propeller raises a TypeError"""
    with raises(TypeError):
        Propeller()

def test_new():
    """Check whether calling new (on ABC) raises a TypeError"""
    with raises(TypeError):
        Propeller.new()