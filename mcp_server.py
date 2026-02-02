import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent

# Token usage callback (works for OpenAI-family models)
from langchain_community.callbacks import get_openai_callback

from mcp.server.transport_security import TransportSecuritySettings

mcp = FastMCP(
    "mcp-sql-agent",
    stateless_http=True,
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    ),
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp.session_manager.run():
        yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],
)
app.mount("/mcp", mcp.streamable_http_app())

# -------------------------
# Build LangChain SQL Agent
# -------------------------
_AGENT = None

ALLOWED_TABLES = ["job_descriptions", "salaries"]
DEFAULT_LIMIT = 10
MAX_LIMIT = 50

print("DB config:", {"MYSQL_HOST": os.environ.get("MYSQL_HOST"), "MYSQL_PORT": os.environ.get("MYSQL_PORT"), "MYSQL_USER": os.environ.get("MYSQL_USER"), "MYSQL_PASSWORD": os.environ.get("MYSQL_PASSWORD"), "MYSQL_DATABASE": os.environ.get("MYSQL_DATABASE")})

def _mysql_uri() -> str:
    host = os.environ.get("MYSQL_HOST")
    port = int(os.environ.get("MYSQL_PORT"))
    user = os.environ.get("MYSQL_USER")
    pwd = os.environ.get("MYSQL_PASSWORD")
    db = os.environ.get("MYSQL_DATABASE")
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}"


def get_llm() -> ChatOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")

    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0")),
        api_key=api_key,
    )


def build_sql_system_prompt(db_name: str) -> str:
    return f"""
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {db_name} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {{top_k}} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(top_k=5)


def get_agent():
    global _AGENT
    if _AGENT is not None:
        return _AGENT

    llm = get_llm()
    db = SQLDatabase.from_uri(_mysql_uri(), include_tables=ALLOWED_TABLES)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    system_prompt = build_sql_system_prompt(db_name=os.environ.get("DB_NAME", "hunter"))

    _AGENT = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="tool-calling",
        system_prompt=system_prompt,
        verbose=False,
        # If you want more observability from the executor, you can set:
        # return_intermediate_steps=True,
    )
    return _AGENT


# -------------
# MCP Tool API
# -------------
class SqlAgentArgs(BaseModel):
    question: str = Field(..., description="Natural language question")
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0


class SqlAgentResponse(BaseModel):
    ok: bool = True
    request_id: str
    model: str
    latency_ms: int

    # ✅ echo inputs for observability
    question: str
    limit: int

    answer: str
    token_usage: Optional[TokenUsage] = None
    error: Optional[str] = None


def _dump_model(m: BaseModel) -> Dict[str, Any]:
    # pydantic v2 -> model_dump; v1 -> dict
    if hasattr(m, "model_dump"):
        return m.model_dump()
    return m.dict()


@mcp.tool()
def sql_agent(args: SqlAgentArgs) -> Dict[str, Any]:
    request_id = str(uuid.uuid4())
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    start = time.perf_counter()
    try:
        agent = get_agent()

        # extra guardrail in the user message
        user_msg = f"{args.question}\n(Use LIMIT <= {args.limit}.)"

        # Capture token usage (must wrap the invoke)
        with get_openai_callback() as cb:
            result = agent.invoke(
                {"input": user_msg},
                {"tags": ["mcp-tool-sql", "sql-agent", "langchain-agent", "sql-tool","v:2.01"]},
            )
        latency_ms = int((time.perf_counter() - start) * 1000)
        answer = (result.get("output") or "").strip()

        resp = SqlAgentResponse(
            ok=True,
            request_id=request_id,
            model=model_name,
            latency_ms=latency_ms,
            question=args.question,
            limit=args.limit,
            answer=answer,
            token_usage=TokenUsage(
                prompt_tokens=getattr(cb, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(cb, "completion_tokens", 0) or 0,
                total_tokens=getattr(cb, "total_tokens", 0) or 0,
                total_cost_usd=float(getattr(cb, "total_cost", 0.0) or 0.0),
            ),
        )
        return _dump_model(resp)

    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        resp = SqlAgentResponse(
            ok=False,
            request_id=request_id,
            model=model_name,
            latency_ms=latency_ms,
            question=args.question,
            limit=args.limit,
            answer="",
            token_usage=None,
            error=f"{type(e).__name__}: {e}",
        )
        return _dump_model(resp)
