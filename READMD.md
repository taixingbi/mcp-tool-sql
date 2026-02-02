```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## run local
```
uvicorn mcp_server:app --reload --port 8000
```

## run docker
```
docker build -t mcp-server .
docker run -p 8000:8000 \
  --env-file .env \
  mcp-server
```

## run fly

### fly
```
brew install flyctl
fly auth login
fly launch
fly deploy
```

input
```
curl -N -sS "https://mcp-tool-sql.fly.dev/mcp/" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc":"2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "sql_agent",
      "arguments": {
        "args": {
          "question": "List 3 job titles in Ventura",
          "limit": 3
        }
      }
    }
  }'
```

### output
```
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\n  \"ok\": true,\n  \"request_id\": \"a3245938-c4a8-4182-8feb-3fa41e212c55\",\n  \"model\": \"gpt-4o-mini\",\n  \"latency_ms\": 17084,\n  \"question\": \"List 3 job titles in Ventura\",\n  \"limit\": 3,\n  \"answer\": \"Here are 3 job titles in Ventura:\\n\\n1. Appraiser Trainee\\n2. Assistant Chief Probation Officer\\n3. Apcd Public Information Specialist\",\n  \"token_usage\": {\n    \"prompt_tokens\": 4327,\n    \"completion_tokens\": 178,\n    \"total_tokens\": 4505,\n    \"total_cost_usd\": 0.00075585\n  },\n  \"error\": null\n}"
      }
    ],
    "structuredContent": {
      "result": {
        "ok": true,
        "request_id": "a3245938-c4a8-4182-8feb-3fa41e212c55",
        "model": "gpt-4o-mini",
        "latency_ms": 17084,
        "question": "List 3 job titles in Ventura",
        "limit": 3,
        "answer": "Here are 3 job titles in Ventura:\n\n1. Appraiser Trainee\n2. Assistant Chief Probation Officer\n3. Apcd Public Information Specialist",
        "token_usage": {
          "prompt_tokens": 4327,
          "completion_tokens": 178,
          "total_tokens": 4505,
          "total_cost_usd": 0.00075585
        },
        "error": null
      }
    },
    "isError": false
  }
}
```

