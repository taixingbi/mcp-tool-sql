"""Tests for mcp_server."""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Patch load_dotenv and env before importing to avoid .env and missing keys
with patch("dotenv.load_dotenv"), patch.dict(os.environ, {
    "OPENAI_API_KEY": "sk-test",
    "MYSQL_HOST": "localhost",
    "MYSQL_PORT": "3306",
    "MYSQL_USER": "test",
    "MYSQL_PASSWORD": "test",
    "MYSQL_DATABASE": "testdb",
}, clear=False):
    # Suppress print during tests
    with patch("builtins.print"):
        from mcp_server import (
        SqlAgentArgs,
        SqlAgentResponse,
        TokenUsage,
        _dump_model,
        _mysql_uri,
        build_sql_system_prompt,
        sql_agent,
    )


class TestSqlAgentArgs:
    """SqlAgentArgs validation."""

    def test_default_limit(self):
        args = SqlAgentArgs(question="test")
        assert args.limit == 10

    def test_valid_limit(self):
        args = SqlAgentArgs(question="test", limit=5)
        assert args.limit == 5

    def test_limit_min_bound(self):
        args = SqlAgentArgs(question="test", limit=1)
        assert args.limit == 1

    def test_limit_max_bound(self):
        args = SqlAgentArgs(question="test", limit=50)
        assert args.limit == 50

    def test_limit_below_min_raises(self):
        with pytest.raises(ValueError):
            SqlAgentArgs(question="test", limit=0)

    def test_limit_above_max_raises(self):
        with pytest.raises(ValueError):
            SqlAgentArgs(question="test", limit=51)


class TestMysqlUri:
    """_mysql_uri with mocked env."""

    @patch.dict(os.environ, {
        "MYSQL_HOST": "localhost",
        "MYSQL_PORT": "3306",
        "MYSQL_USER": "test",
        "MYSQL_PASSWORD": "test",
        "MYSQL_DATABASE": "testdb",
    }, clear=False)
    def test_mysql_uri_format(self):
        uri = _mysql_uri()
        assert uri == "mysql+pymysql://test:test@localhost:3306/testdb"


class TestBuildSqlSystemPrompt:
    """build_sql_system_prompt."""

    def test_includes_db_name(self):
        prompt = build_sql_system_prompt(db_name="mydb")
        assert "mydb" in prompt

    def test_includes_top_k(self):
        prompt = build_sql_system_prompt(db_name="x")
        assert "5" in prompt  # top_k=5

    def test_no_dml_instruction(self):
        prompt = build_sql_system_prompt(db_name="x")
        assert "INSERT" in prompt or "DML" in prompt
        assert "DO NOT" in prompt or "Never" in prompt


class TestDumpModel:
    """_dump_model for Pydantic v1/v2."""

    def test_dumps_to_dict(self):
        args = SqlAgentArgs(question="q", limit=5)
        d = _dump_model(args)
        assert isinstance(d, dict)
        assert d["question"] == "q"
        assert d["limit"] == 5


class TestSqlAgent:
    """sql_agent tool with mocked agent and callback."""

    @patch("mcp_server.get_openai_callback")
    @patch("mcp_server.get_agent")
    def test_success_returns_ok_response(self, mock_get_agent, mock_callback):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Here are 3 job titles."}

        mock_cb = MagicMock()
        mock_cb.prompt_tokens = 100
        mock_cb.completion_tokens = 50
        mock_cb.total_tokens = 150
        mock_cb.total_cost = 0.001
        mock_callback.return_value.__enter__ = MagicMock(return_value=mock_cb)
        mock_callback.return_value.__exit__ = MagicMock(return_value=False)

        mock_get_agent.return_value = mock_agent

        args = SqlAgentArgs(question="List 5 jobs", limit=5)
        result = sql_agent(args)

        assert result["ok"] is True
        assert result["answer"] == "Here are 3 job titles."
        assert result["question"] == "List 5 jobs"
        assert result["limit"] == 5
        assert result["error"] is None
        assert "request_id" in result
        assert result["token_usage"]["prompt_tokens"] == 100
        assert result["token_usage"]["completion_tokens"] == 50

    @patch("mcp_server.get_agent")
    def test_exception_returns_error_response(self, mock_get_agent):
        mock_get_agent.side_effect = ValueError("OPENAI_API_KEY is required")

        args = SqlAgentArgs(question="test", limit=5)
        result = sql_agent(args)

        assert result["ok"] is False
        assert result["answer"] == ""
        assert "ValueError" in result["error"]
        assert result["token_usage"] is None
