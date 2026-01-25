def test_eval_runner_smoke():
    # This is a smoke test: if it can import tools/router, we're good.
    from src.agent.router import RuleBasedRouter
    from src.agent import tools

    r = RuleBasedRouter()
    rr = r.route("sales in 2013")
    assert rr.route.value in ("data", "rag", "mixed", "general")
