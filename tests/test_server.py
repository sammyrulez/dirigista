from dirigista.server import mcp


async def _tool(name: str):
    return next(tool for tool in await mcp.list_tools() if tool.name == name)


async def test_server_exposes_the_three_classification_tools():
    tools = await mcp.list_tools()

    assert {tool.name for tool in tools} == {"verify", "categorize", "score"}


async def test_verify_takes_a_statement_and_an_optional_context():
    schema = (await _tool("verify")).input_schema

    assert schema["properties"]["statement"]["type"] == "string"
    assert schema["required"] == ["statement"]


async def test_verify_reports_a_three_valued_verdict_with_confidence():
    schema = (await _tool("verify")).output_schema

    assert set(schema["properties"]) == {"verdict", "probability"}
    assert schema["properties"]["verdict"]["enum"] == ["true", "false", "uncertain"]


async def test_categorize_takes_the_candidate_categories_from_the_caller():
    schema = (await _tool("categorize")).input_schema

    assert schema["properties"]["categories"]["items"] == {"type": "string"}
    assert set(schema["required"]) == {"statement", "categories"}


async def test_categorize_reports_the_chosen_category_with_confidence():
    schema = (await _tool("categorize")).output_schema

    assert set(schema["properties"]) == {"category", "confidence", "probabilities"}


async def test_score_takes_the_ordinal_scale_from_the_caller():
    schema = (await _tool("score")).input_schema

    assert schema["properties"]["criteria"]["items"] == {"type": "string"}
    assert set(schema["required"]) == {"statement", "criteria"}


async def test_score_reports_the_chosen_criterion_and_its_position_on_the_scale():
    schema = (await _tool("score")).output_schema

    assert set(schema["properties"]) == {"criterion", "level", "confidence"}
    assert schema["properties"]["level"]["type"] == "integer"
    assert schema["properties"]["level"]["minimum"] == 1
