# tests/test_router.py
from src.agent.router import RuleBasedRouter, Route


def test_router_basic():
    r = RuleBasedRouter()

    # --- DATA ---
    assert r.route("Which month sales maximum in 2016").route == Route.DATA
    assert r.route("Show daily sales for store 1 family Grocery I in jan 2013").route == Route.DATA
    assert r.route("Plot sales trend for 2015").route == Route.DATA
    assert r.route("Compare sales by family in 2014").route == Route.DATA
    assert r.route("2016-01-01 to 2016-12-31 total sales").route == Route.DATA

    # --- RAG ---
    assert r.route("According to the document, what is Industry 4.0?").route == Route.RAG
    assert r.route("From the PDF, cite sources about automation").route == Route.RAG
    assert r.route("Show evidence and citations from the report").route == Route.RAG

    # --- MIXED ---
    assert r.route("From the document, compare sales trends in 2016").route == Route.MIXED
    assert r.route("Cite the PDF and plot sales by month").route == Route.MIXED


if __name__ == "__main__":
    test_router_basic()
    print("OK - router tests passed")
