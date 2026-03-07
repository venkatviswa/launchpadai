"""Generate the UI layer — test interfaces for the agent."""
from pathlib import Path


def generate_ui_layer(config: dict, project_path: Path):
    ui = config["ui"]
    if ui == "none":
        return

    if ui == "streamlit":
        _generate_streamlit(config, project_path)
    elif ui == "gradio":
        _generate_gradio(config, project_path)
    elif ui == "nextjs":
        _generate_nextjs(config, project_path)


def _generate_streamlit(config: dict, project_path: Path):
    """Generate a Streamlit chat UI with optional auth."""
    auth = config.get("auth", "none")

    auth_import = ""
    auth_check = ""
    auth_sidebar = ""

    if auth in ("simple", "multi_user"):
        auth_import = "from auth.provider import auth_provider"
        username_field = ""
        if auth == "multi_user":
            username_field = """
        username = st.text_input("Username", key="login_user")"""

        auth_check = f'''
# === Authentication ===
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.auth_user = None
    st.session_state.auth_token = None

if not st.session_state.authenticated:
    st.markdown("### 🔐 Login")
    with st.form("login_form"):{username_field}
        password = st.text_input("Password", type="password", key="login_pass")
        submitted = st.form_submit_button("Login")

        if submitted:
            result = auth_provider.authenticate(
                username={'"username"' if auth == 'multi_user' else 'None'},
                password=password,
            )
            if result["authenticated"]:
                st.session_state.authenticated = True
                st.session_state.auth_user = result["user"]
                st.session_state.auth_token = result["token"]
                st.rerun()
            else:
                st.error(result.get("error", "Authentication failed"))
    st.stop()
'''

        auth_sidebar = '''
    st.markdown(f"**User:** {st.session_state.auth_user}")
    if st.button("Logout"):
        auth_provider.logout(st.session_state.auth_token)
        st.session_state.authenticated = False
        st.session_state.auth_user = None
        st.session_state.auth_token = None
        st.rerun()'''

    _write(project_path / "ui" / "app.py", f'''"""Streamlit Chat UI for {config['project_name']}."""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import agent
{auth_import}

# Page config
st.set_page_config(
    page_title="{config['project_name']}",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 {config['project_name']}")
st.caption("{config['agent_description']}")
{auth_check}
# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Show user message
    st.session_state.messages.append({{"role": "user", "content": prompt}})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent.run(prompt, session_id=st.session_state.session_id)
                response = result["response"] if isinstance(result, dict) else str(result)
            except Exception as e:
                response = f"Error: {{e}}"

        st.markdown(response)
        st.session_state.messages.append({{"role": "assistant", "content": response}})

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    st.text(f"Session: {{st.session_state.session_id}}")
{auth_sidebar}
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("Built with [LaunchpadAI](https://github.com/launchpadai/launchpadai)")
''')


def _generate_gradio(config: dict, project_path: Path):
    """Generate a Gradio chat UI with optional auth."""
    auth = config.get("auth", "none")

    # Gradio has built-in auth support
    auth_arg = ""
    if auth == "simple":
        auth_arg = """
    # Simple auth: password only (use any username)
    import os
    password = os.getenv("APP_PASSWORD", "changeme")
    launch_kwargs["auth"] = ("user", password)
    launch_kwargs["auth_message"] = "Enter password (username can be anything)"
"""
    elif auth == "multi_user":
        auth_arg = """
    # Multi-user auth from env var
    import os
    users_str = os.getenv("APP_USERS", "admin:changeme")
    auth_pairs = []
    for pair in users_str.split(","):
        if ":" in pair:
            u, p = pair.strip().split(":", 1)
            auth_pairs.append((u.strip(), p.strip()))
    if auth_pairs:
        launch_kwargs["auth"] = auth_pairs
        launch_kwargs["auth_message"] = "Login to access the agent"
"""

    _write(project_path / "ui" / "app.py", f'''"""Gradio Chat UI for {config['project_name']}."""
import gradio as gr
import sys
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import agent


def respond(message: str, history: list, session_id: str):
    """Process a message and return the response."""
    try:
        result = agent.run(message, session_id=session_id)
        response = result["response"] if isinstance(result, dict) else str(result)
    except Exception as e:
        response = f"Error: {{e}}"
    return response


# Build the UI
with gr.Blocks(title="{config['project_name']}") as demo:
    gr.Markdown("# 🤖 {config['project_name']}")
    gr.Markdown("{config['agent_description']}")

    session_id = gr.State(value=lambda: str(uuid.uuid4())[:8])

    chatbot = gr.ChatInterface(
        fn=respond,
        additional_inputs=[session_id],
        examples=[
            "Hello! What can you help me with?",
            "Tell me about yourself.",
        ],
        title=None,
    )

    gr.Markdown("---")
    gr.Markdown("Built with [LaunchpadAI](https://github.com/launchpadai/launchpadai)")


if __name__ == "__main__":
    launch_kwargs = {{"server_port": 7860}}
{auth_arg}
    demo.launch(**launch_kwargs)
''')


def _generate_nextjs(config: dict, project_path: Path):
    """Generate a Next.js frontend with API backend."""
    ui_path = project_path / "ui"

    # package.json
    _write(ui_path / "package.json", '''{
  "name": "''' + config["project_name"] + '''-ui",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "next": "^14.2.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0"
  },
  "devDependencies": {
    "tailwindcss": "^3.4.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0"
  }
}
''')

    # Main page
    _write(ui_path / "app" / "page.js", f'''\"use client\";
import {{ useState, useRef, useEffect }} from "react";

export default function Chat() {{
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEnd = useRef(null);

  const scrollToBottom = () => {{
    messagesEnd.current?.scrollIntoView({{ behavior: "smooth" }});
  }};

  useEffect(() => {{
    scrollToBottom();
  }}, [messages]);

  const sendMessage = async () => {{
    if (!input.trim() || loading) return;

    const userMsg = {{ role: "user", content: input }};
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {{
      const res = await fetch("http://localhost:8000/chat", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ message: input, session_id: "default" }}),
      }});
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {{ role: "assistant", content: data.response }},
      ]);
    }} catch (err) {{
      setMessages((prev) => [
        ...prev,
        {{ role: "assistant", content: "Error connecting to the agent API." }},
      ]);
    }}
    setLoading(false);
  }};

  return (
    <div style={{{{ maxWidth: 700, margin: "0 auto", padding: 20, fontFamily: "sans-serif" }}}}>
      <h1>🤖 {config['project_name']}</h1>
      <p style={{{{ color: "#666" }}}}>{config['agent_description']}</p>

      <div style={{{{ border: "1px solid #ddd", borderRadius: 8, padding: 16, minHeight: 400, maxHeight: 600, overflowY: "auto", marginBottom: 16 }}}}>
        {{messages.map((msg, i) => (
          <div key={{i}} style={{{{ marginBottom: 12, textAlign: msg.role === "user" ? "right" : "left" }}}}>
            <div style={{{{
              display: "inline-block",
              padding: "8px 14px",
              borderRadius: 12,
              background: msg.role === "user" ? "#007AFF" : "#f0f0f0",
              color: msg.role === "user" ? "white" : "black",
              maxWidth: "80%",
              textAlign: "left",
            }}}}>
              {{msg.content}}
            </div>
          </div>
        ))}}
        {{loading && <div style={{{{ color: "#999" }}}}>Thinking...</div>}}
        <div ref={{messagesEnd}} />
      </div>

      <div style={{{{ display: "flex", gap: 8 }}}}>
        <input
          value={{input}}
          onChange={{(e) => setInput(e.target.value)}}
          onKeyDown={{(e) => e.key === "Enter" && sendMessage()}}
          placeholder="Ask me anything..."
          style={{{{ flex: 1, padding: "10px 14px", borderRadius: 8, border: "1px solid #ddd", fontSize: 16 }}}}
        />
        <button
          onClick={{sendMessage}}
          disabled={{loading}}
          style={{{{ padding: "10px 20px", borderRadius: 8, background: "#007AFF", color: "white", border: "none", cursor: "pointer", fontSize: 16 }}}}
        >
          Send
        </button>
      </div>
    </div>
  );
}}
''')

    # Layout
    _write(ui_path / "app" / "layout.js", f'''export const metadata = {{
  title: "{config['project_name']}",
  description: "{config['agent_description']}",
}};

export default function RootLayout({{ children }}) {{
  return (
    <html lang="en">
      <body>{{children}}</body>
    </html>
  );
}}
''')


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
