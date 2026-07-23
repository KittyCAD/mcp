import json
from pathlib import Path

from zoo_mcp.api_docs.index import ZooApiDocsIndex


def _result_id(result: dict) -> str:
    return str(result.get("operation_id") or result.get("guide_id"))


def test_representative_questions_recall_expected_result_in_top_three():
    cases = json.loads(
        (Path(__file__).parent / "data" / "api_docs_retrieval_eval.json").read_text()
    )
    index = ZooApiDocsIndex.load_bundled()
    successful = 0

    for case in cases:
        results = index.search(case["question"], limit=3)["results"]
        result_ids = {_result_id(result) for result in results}
        expected = set(case["expected"])
        if not expected:
            assert results == [], case["question"]
            successful += 1
        elif result_ids.intersection(expected):
            successful += 1

    assert successful / len(cases) >= 0.90


def test_retrieval_eval_has_required_coverage_and_exclusion_compliance():
    cases = json.loads(
        (Path(__file__).parent / "data" / "api_docs_retrieval_eval.json").read_text()
    )
    index = ZooApiDocsIndex.load_bundled()

    assert len(cases) >= 30
    for query in ("get_ipinfo", "internal_get_api_token_for_discord_user"):
        result_ids = {
            _result_id(result)
            for result in index.search(query, limit=10, include_deprecated=True)[
                "results"
            ]
        }
        assert query not in result_ids
        assert "error" in index.get_operation(query)
