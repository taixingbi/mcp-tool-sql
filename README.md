# mcp-tool-sql
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


