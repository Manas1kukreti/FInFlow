# Railway Deployment

Deploy the frontend, backend, and agent service as separate Railway services.

## Backend service

Set the service root to `backend` so Railway uses `backend/Dockerfile`.

Required variables:

```env
ENVIRONMENT=production
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}
AGENT_BASE_URL=https://<your-agent-service-domain>
BACKEND_CALLBACK_URL=https://<your-backend-domain>/api/agent/callback
AGENT_CALLBACK_SECRET=<strong-shared-secret>
UPLOAD_DIR=/app/storage/uploads
OUTPUT_DIR=/app/storage/outputs
JWT_SECRET_KEY=<generate-a-random-64-character-secret>
CORS_ORIGINS=https://<your-frontend-domain>
FRONTEND_BASE_URL=https://<your-frontend-domain>
DEFAULT_ADMIN_EMAIL=<your-admin-email>
DEFAULT_ADMIN_PASSWORD=<strong-one-time-admin-password>
DEFAULT_ADMIN_NAME=FinFlow Admin
AGENT_EMAIL=<optional-agent-email>
AGENT_PASSWORD=<optional-agent-password>
AGENT_NAME=FinFlow Orchestrator
EMAILS_ENABLED=false
```

Optional email variables:

```env
EMAILS_ENABLED=true
SMTP_HOST=<smtp-host>
SMTP_PORT=587
SMTP_USERNAME=<smtp-username>
SMTP_PASSWORD=<smtp-password>
SMTP_FROM_EMAIL=<verified-sender-email>
SMTP_TLS=true
```

The backend listens on Railway's injected `PORT` variable and runs Alembic migrations before starting.

If logs mention `localhost:5433`, the backend service is still pointed at a local database URL instead of Railway Postgres.

## Agent service

Set the service root to `agent-framework`.

Required variables:

```env
ENVIRONMENT=production
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}
BACKEND_CALLBACK_URL=https://<your-backend-domain>/api/agent/callback
AGENT_CALLBACK_SECRET=<same-shared-secret-as-backend>
UPLOAD_DIR=/app/storage/uploads
GROQ_API_KEY=<your-groq-key>
OUTPUT_DIR=/app/outputs
HOST=0.0.0.0
PORT=8001
```

The agent service must be able to reach the backend callback URL and the database.

## Frontend service

Set the service root to `frontend` so Railway uses `frontend/Dockerfile`.

Build variable:

```env
VITE_API_BASE_URL=https://<your-backend-domain>
```

The frontend container serves on port `80`. Keep the public domain pointed there.

## Agent test endpoints

Use the deployed backend URLs:

```text
POST https://<your-backend-domain>/api/agent/login
POST https://<your-backend-domain>/api/agent/upload
```

If `AGENT_EMAIL` and `AGENT_PASSWORD` are set, the backend seeds a dedicated employee account for that flow on startup.

## Security checklist

- Do not commit `.env` files or real credentials.
- Store all production secrets in Railway variables.
- Use exact frontend origins in `CORS_ORIGINS`.
- Rotate `DEFAULT_ADMIN_PASSWORD` after first login.
- Give external posting agents their own employee account.
- Keep supported upload types aligned between the frontend and backend parser.

