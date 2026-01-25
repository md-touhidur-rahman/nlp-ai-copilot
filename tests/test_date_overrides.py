# tests/test_date_overrides.py
from src.agent import tools


def test_month_range_year_last():
    q = "sales from march to june in 2013 at store 2"
    mr = tools.extract_month_range(q)
    assert mr == ("2013-03-01", "2013-06-30")


def test_month_range_year_first():
    q = "sales between 2013 may to october"
    mr = tools.extract_month_range(q)
    assert mr == ("2013-05-01", "2013-10-31")


def test_month_range_between_year_first_and():
    q = "between 2013 may and october sales"
    mr = tools.extract_month_range(q)
    assert mr == ("2013-05-01", "2013-10-31")


def test_month_range_between_year_last():
    q = "sales between may and october 2013"
    mr = tools.extract_month_range(q)
    assert mr == ("2013-05-01", "2013-10-31")


def test_year_range_from_to():
    q = "sales from 2014 to 2016"
    yr = tools.extract_year_range(q)
    assert yr == ("2014-01-01", "2016-12-31")


def test_year_range_between_and():
    q = "sales between 2014 and 2016"
    yr = tools.extract_year_range(q)
    assert yr == ("2014-01-01", "2016-12-31")


def test_months_without_year_should_refuse():
    # This checks the guard logic via the helper, not the full run_data() (which needs dataset)
    class DummyDQ:
        start = None
        end = None

    assert tools.months_present_without_year("sales in store 2 in january to march", DummyDQ()) is True
    assert tools.months_present_without_year("sales in store 2 in january to march 2013", DummyDQ()) is False
