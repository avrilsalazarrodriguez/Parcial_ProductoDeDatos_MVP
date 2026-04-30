from frontend.friendly_display import clean_display_value, first_int_from_label


def test_clean_display_value_handles_none_strings():
    assert clean_display_value(None) == "Sin clasificar"
    assert clean_display_value("None") == "Sin clasificar"
    assert clean_display_value("unknown") == "Sin clasificar"
    assert clean_display_value("Books") == "Books"


def test_first_int_from_label():
    assert first_int_from_label("2 — Some shop") == 2
    assert first_int_from_label("123 — Producto") == 123
    assert first_int_from_label("Todos") is None
