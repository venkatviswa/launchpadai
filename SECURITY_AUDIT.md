# Security Audit Report — LaunchpadAI

**Date:** 2026-03-08
**Scope:** Full codebase security review
**Standards:** OWASP Top 10, OWASP LLM Top 10, OWASP Agentic AI Security Guidelines

---

## Executive Summary

LaunchpadAI is a CLI scaffolding tool that generates production-ready agentic AI applications. This audit reviews both the tool itself and the **generated code templates** it produces, since vulnerabilities in templates propagate to every generated project.

**Findings:** 14 security issues identified (2 Critical, 3 High, 7 Medium, 2 Low).
**Fixes Applied:** 8 issues fixed in this audit. Remaining items documented as recommendations.

---

## Findings Summary

| # | Severity | Category | Issue | Status |
|---|----------|----------|-------|--------|
| 1 | **CRITICAL** | OWASP A05 | Wildcard CORS with credentials | **FIXED** |
| 2 | **CRITICAL** | OWASP LLM01 | Weak prompt injection guardrails | **FIXED** |
| 3 | **HIGH** | OWASP A07 | Broken password comparison (not constant-time) | **FIXED** |
| 4 | **HIGH** | OWASP A07 | No brute-force protection on auth | **FIXED** |
| 5 | **HIGH** | OWASP LLM06 | No PII detection on LLM output | **FIXED** |
| 6 | **MEDIUM** | OWASP A03 | No input length validation on API | **FIXED** |
| 7 | **MEDIUM** | OWASP A01 | Path traversal in document loader | **FIXED** |
| 8 | **MEDIUM** | OWASP A05 | Missing security headers | **FIXED** |
| 9 | **MEDIUM** | OWASP A05 | Docker container runs as root | **FIXED** |
| 10 | **MEDIUM** | OWASP A07 | Hardcoded DB credentials in Docker Compose | **FIXED** |
| 11 | **MEDIUM** | Agentic AI | No tool existence check before execution | **FIXED** |
| 12 | **MEDIUM** | OWASP A04 | No rate limiting on API endpoints | Recommendation |
| 13 | **LOW** | OWASP A09 | Error details exposed in logs | Recommendation |
| 14 | **LOW** | OWASP A06 | No dependency vulnerability scanning | Recommendation |

---

## Detailed Findings & Fixes

### 1. CRITICAL — Wildcard CORS with Credentials (OWASP A05: Security Misconfiguration)

**File:** `generators/api_layer.py`

**Before:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Any origin
    allow_credentials=True,       # With credentials!
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Risk:** Allows any website to make authenticated cross-origin requests to the API, enabling CSRF attacks and credential theft.

**Fix Applied:** Restricted CORS to configurable allowed origins (defaults to localhost), limited methods and headers:
```python
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
```

---

### 2. CRITICAL — Weak Prompt Injection Guardrails (OWASP LLM01: Prompt Injection)

**File:** `generators/guardrails_layer.py`

**Before:** Only 6 simple string-matching patterns that are trivially bypassed with variations, typos, or encoding tricks.

**Risk:** Attackers can override system prompts, extract instructions, or manipulate agent behavior.

**Fix Applied:**
- Expanded from 6 patterns to 17+ regex-based detection rules
- Added detection for delimiter injection (`[INST]`, `<|system|>`, etc.)
- Added system prompt extraction detection
- Added base64 encoding evasion detection
- Added input length limits (>50k characters flagged)
- Added special character ratio analysis
- Pre-compiled patterns for performance

---

### 3. HIGH — Broken Password Comparison (OWASP A07: Identification & Authentication Failures)

**File:** `generators/auth_layer.py`

**Before:**
```python
def hmac_compare(a: bytes, b: bytes) -> bool:
    return hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()
```

**Risk:** Despite the function name, this is NOT constant-time comparison. Python string `==` operator short-circuits, making it vulnerable to timing attacks.

**Fix Applied:** Replaced with `hmac.compare_digest()` from the standard library, which is guaranteed constant-time:
```python
hmac.compare_digest((password or "").encode(), expected.encode())
```

---

### 4. HIGH — No Brute-Force Protection (OWASP A07)

**File:** `generators/auth_layer.py`

**Before:** Unlimited authentication attempts with no lockout mechanism.

**Fix Applied:** Added rate limiting with lockout:
- Tracks failed attempts per user/identifier
- Locks out after 10 failed attempts
- 5-minute lockout window
- Automatic lockout expiry
- Logging of lockout events

---

### 5. HIGH — No PII Detection on LLM Output (OWASP LLM06: Sensitive Information Disclosure)

**File:** `generators/guardrails_layer.py`

**Before:** Output filter had a `# TODO: Add PII detection on output` comment with no implementation.

**Fix Applied:** Added regex-based PII detection and automatic redaction for:
- Email addresses
- Phone numbers
- Social Security Numbers
- Credit card numbers
- System prompt leak patterns

Detected PII is automatically replaced with `[REDACTED_TYPE]` tokens.

---

### 6. MEDIUM — No Input Length Validation (OWASP A03: Injection)

**File:** `generators/api_layer.py`

**Before:**
```python
class ChatRequest(BaseModel):
    message: str  # No length limit
```

**Risk:** Attackers can send extremely large messages, causing resource exhaustion or exploiting LLM token limits.

**Fix Applied:**
```python
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = Field("default", max_length=128)
```

---

### 7. MEDIUM — Path Traversal in Document Loader (OWASP A01: Broken Access Control)

**File:** `generators/knowledge_layer.py`

**Before:** `rglob("*")` with no path boundary checks, potentially loading files outside the intended directory via symlinks.

**Fix Applied:**
- Added `file.resolve().relative_to(dir_path)` check to prevent traversal
- Added 50MB file size limit to prevent resource exhaustion
- Replaced `print()` with `logging.warning()` for security event tracking

---

### 8. MEDIUM — Missing Security Headers (OWASP A05: Security Misconfiguration)

**File:** `generators/api_layer.py`

**Before:** No security headers set on API responses.

**Fix Applied:** Added security headers middleware:
- `X-Content-Type-Options: nosniff` — prevents MIME sniffing
- `X-Frame-Options: DENY` — prevents clickjacking
- `X-XSS-Protection: 1; mode=block` — XSS filter
- `Referrer-Policy: strict-origin-when-cross-origin` — controls referrer leakage
- `Permissions-Policy` — restricts camera, microphone, geolocation

---

### 9. MEDIUM — Docker Container Runs as Root (OWASP A05)

**File:** `generators/docker_layer.py`

**Before:** No `USER` directive; container runs as root.

**Fix Applied:** Added non-root user:
```dockerfile
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser
RUN chown -R appuser:appuser /app
USER appuser
```

---

### 10. MEDIUM — Hardcoded Database Credentials (OWASP A07)

**File:** `generators/docker_layer.py`

**Before:** `POSTGRES_PASSWORD: agent` hardcoded in docker-compose.

**Fix Applied:** Changed to environment variable references with required password:
```yaml
POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD in .env}
```

Also fixed Weaviate anonymous access from `"true"` to configurable with API key authentication.

---

### 11. MEDIUM — No Tool Existence Check Before Execution (Agentic AI Security)

**File:** `generators/agents_layer.py`

**Before:** Tool name from LLM response was passed directly to `registry.execute()`, which throws a `ValueError` with internal details.

**Fix Applied:**
- Added `has_tool()` method to ToolRegistry for safe pre-execution validation
- Check tool existence before attempting execution
- Error messages don't expose internal exception details (returns `type(e).__name__` only)

---

## Recommendations (Not Yet Implemented)

### 12. MEDIUM — Add Rate Limiting to API Endpoints

Currently no request rate limiting exists. Recommend adding:
```python
# pip install slowapi
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: ChatRequest): ...
```

### 13. LOW — Structured Logging with Sensitive Data Filtering

Replace `print()` statements with structured logging throughout generated code. Add log filtering to prevent API keys, tokens, and credentials from being logged.

### 14. LOW — Dependency Vulnerability Scanning

Add dependency scanning to the development workflow:
```bash
pip install pip-audit safety
pip-audit                    # Check for known vulnerabilities
safety check                 # Alternative scanner
```

---

## OWASP Agentic AI Compliance Summary

| OWASP Agentic AI Principle | Status |
|-----------------------------|--------|
| **Input Validation** — Validate all user inputs before LLM processing | Implemented (guardrails + API validation) |
| **Prompt Injection Defense** — Detect and block injection attempts | Implemented (regex patterns, delimiter detection) |
| **Output Sanitization** — Filter LLM outputs for safety | Implemented (PII redaction, hallucination detection) |
| **Tool Execution Safety** — Validate tools before execution | Implemented (has_tool check, error sanitization) |
| **Agent Loop Limits** — Prevent runaway agent loops | Implemented (max_iterations=10 default) |
| **Least Privilege for Tools** — Tools registered explicitly | Implemented (ToolRegistry pattern) |
| **Human-in-the-Loop** — Approval before critical actions | Partial (auth required, no approval workflow) |
| **Cost Controls** — Monitor and limit LLM spending | Partial (cost tracking, no hard caps) |
| **Observability** — Trace and log agent actions | Implemented (LangFuse/LangSmith/OpenTelemetry) |
| **Memory Isolation** — Session-scoped conversation history | Implemented (session_id-based isolation) |

---

## Files Modified

| File | Changes |
|------|---------|
| `generators/api_layer.py` | CORS restriction, security headers, input validation |
| `generators/auth_layer.py` | Constant-time comparison, brute-force protection, logging |
| `generators/guardrails_layer.py` | Expanded prompt injection patterns, PII output redaction |
| `generators/knowledge_layer.py` | Path traversal protection, file size limits |
| `generators/docker_layer.py` | Non-root user, credential externalization |
| `generators/agents_layer.py` | Tool existence validation, error sanitization |
| `generators/tools_layer.py` | Added `has_tool()` method |
| `generators/agents_layer.py` | Added AgentScript framework with Salesforce OAuth2 auth, session-scoped API calls |

---

## Salesforce AgentScript Security Notes

The AgentScript framework integration follows Salesforce security best practices:

- **OAuth2 Client Credentials** — Agent API authentication uses the standard Salesforce connected-app client credentials flow (no hardcoded tokens)
- **Environment-based secrets** — `SF_CLIENT_ID`, `SF_CLIENT_SECRET`, `SF_INSTANCE_URL`, and `SF_AGENT_ID` are loaded from environment variables, never committed to code
- **Session isolation** — Each conversation uses a separate Agentforce session via `externalSessionKey`, preventing cross-session data leakage
- **Declarative safety** — AgentScript's `.agent` DSL files include guardrail instructions directly in the reasoning block when guardrails are enabled
- **No raw credential storage** — Access tokens are obtained at runtime and held only in memory

---

*This audit was performed against the LaunchpadAI codebase. For production deployments of generated projects, additional security measures (WAF, HTTPS enforcement, secrets management, penetration testing) are strongly recommended.*
