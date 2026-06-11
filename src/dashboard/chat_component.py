import streamlit.components.v1 as components
import sys
import os


def get_rag_context(project_root: str) -> str:
    try:
        sys.path.insert(0, project_root)
        from src.rag.retriever import retrieve_context

        questions = [
            "model performance metrics",
            "data drift features",
            "retraining recommendation",
            "feature descriptions"
        ]
        docs = []
        for q in questions:
            docs.extend(retrieve_context(q, n_results=2))

        seen   = set()
        unique = []
        for d in docs:
            if d not in seen:
                seen.add(d)
                unique.append(d)

        context = "\n\n---\n\n".join(unique[:6])
        context = context.replace("\\", "\\\\").replace("`", "\\`").replace("'", "\\'")
        return context
    except Exception as e:
        return f"Context unavailable: {str(e)}"


def render_floating_chat(project_root: str):
    context = get_rag_context(project_root)

    # This iframe injects the floating widget into the parent
    # Streamlit page using window.parent.document
    # Same-origin policy allows this since both run on localhost
    html = f"""
<script>
(function() {{
  const parent = window.parent.document;

  // Avoid injecting twice on Streamlit reruns
  if (parent.getElementById('mlops-chat-bubble')) return;

  const RAG_CONTEXT = `{context}`;
  const OLLAMA_URL  = 'http://localhost:11434/api/generate';
  const MODEL       = 'llama3.2';

  // Inject styles into parent page
  const style = parent.createElement('style');
  style.id    = 'mlops-chat-styles';
  style.textContent = `
    #mlops-chat-bubble {{
      position: fixed !important;
      bottom: 28px !important;
      right: 28px !important;
      width: 54px !important;
      height: 54px !important;
      background: #1976d2 !important;
      border-radius: 50% !important;
      cursor: pointer !important;
      z-index: 999999 !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      box-shadow: 0 4px 16px rgba(25,118,210,0.45) !important;
      border: none !important;
      transition: transform 0.2s, box-shadow 0.2s !important;
    }}
    #mlops-chat-bubble:hover {{
      transform: scale(1.08) !important;
      box-shadow: 0 6px 22px rgba(25,118,210,0.55) !important;
    }}
    #mlops-chat-panel {{
      position: fixed !important;
      bottom: 94px !important;
      right: 28px !important;
      width: 340px !important;
      height: 460px !important;
      background: #1e1e2e !important;
      border: 1px solid #333 !important;
      border-radius: 14px !important;
      box-shadow: 0 12px 40px rgba(0,0,0,0.45) !important;
      z-index: 999998 !important;
      display: none !important;
      flex-direction: column !important;
      overflow: hidden !important;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }}
    #mlops-chat-panel.mlops-open {{
      display: flex !important;
    }}
    #mlops-header {{
      padding: 13px 16px !important;
      background: #1976d2 !important;
      color: white !important;
      font-size: 14px !important;
      font-weight: 500 !important;
      display: flex !important;
      justify-content: space-between !important;
      align-items: center !important;
      border-radius: 14px 14px 0 0 !important;
    }}
    #mlops-header .sub {{
      font-size: 11px !important;
      opacity: 0.8 !important;
    }}
    #mlops-close {{
      background: none !important;
      border: none !important;
      color: white !important;
      cursor: pointer !important;
      font-size: 18px !important;
      opacity: 0.8 !important;
      padding: 0 4px !important;
      line-height: 1 !important;
    }}
    #mlops-close:hover {{ opacity: 1 !important; }}
    #mlops-messages {{
      flex: 1 !important;
      overflow-y: auto !important;
      padding: 12px !important;
      display: flex !important;
      flex-direction: column !important;
      gap: 10px !important;
      scrollbar-width: thin !important;
      scrollbar-color: #444 transparent !important;
    }}
    .mlops-msg {{
      max-width: 85% !important;
      padding: 9px 12px !important;
      border-radius: 12px !important;
      font-size: 13px !important;
      line-height: 1.5 !important;
      word-wrap: break-word !important;
      color: white !important;
    }}
    .mlops-msg.user {{
      background: #1976d2 !important;
      align-self: flex-end !important;
      border-bottom-right-radius: 4px !important;
    }}
    .mlops-msg.bot {{
      background: #2a2a3e !important;
      color: #e0e0e0 !important;
      align-self: flex-start !important;
      border-bottom-left-radius: 4px !important;
      border: 1px solid #333 !important;
    }}
    .mlops-msg.typing {{
      background: #2a2a3e !important;
      color: #888 !important;
      align-self: flex-start !important;
      border: 1px solid #333 !important;
      border-bottom-left-radius: 4px !important;
    }}
    #mlops-quick {{
      padding: 8px 12px 4px !important;
      display: flex !important;
      gap: 6px !important;
      flex-wrap: wrap !important;
    }}
    .mlops-qbtn {{
      background: #2a2a3e !important;
      border: 1px solid #444 !important;
      color: #aaa !important;
      padding: 5px 10px !important;
      border-radius: 20px !important;
      font-size: 11px !important;
      cursor: pointer !important;
      transition: all 0.2s !important;
      white-space: nowrap !important;
    }}
    .mlops-qbtn:hover {{
      background: #1976d2 !important;
      color: white !important;
      border-color: #1976d2 !important;
    }}
    #mlops-input-row {{
      display: flex !important;
      gap: 8px !important;
      padding: 10px 12px !important;
      border-top: 1px solid #333 !important;
      background: #1e1e2e !important;
      border-radius: 0 0 14px 14px !important;
    }}
    #mlops-input {{
      flex: 1 !important;
      background: #2a2a3e !important;
      border: 1px solid #444 !important;
      border-radius: 8px !important;
      color: #e0e0e0 !important;
      font-size: 13px !important;
      padding: 8px 10px !important;
      outline: none !important;
      height: 36px !important;
    }}
    #mlops-input:focus {{ border-color: #1976d2 !important; }}
    #mlops-input::placeholder {{ color: #666 !important; }}
    #mlops-send {{
      background: #1976d2 !important;
      border: none !important;
      border-radius: 8px !important;
      color: white !important;
      width: 36px !important;
      height: 36px !important;
      cursor: pointer !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      flex-shrink: 0 !important;
      font-size: 16px !important;
    }}
    #mlops-send:hover {{ background: #1565c0 !important; }}
    #mlops-send:disabled {{ background: #444 !important; cursor: not-allowed !important; }}
    @keyframes mlops-dot {{
      0%, 80%, 100% {{ opacity: 0.2; transform: scale(0.8); }}
      40% {{ opacity: 1; transform: scale(1); }}
    }}
    .mlops-dots span {{
      display: inline-block !important;
      width: 6px !important; height: 6px !important;
      background: #888 !important;
      border-radius: 50% !important;
      margin: 0 2px !important;
      animation: mlops-dot 1.2s infinite !important;
    }}
    .mlops-dots span:nth-child(2) {{ animation-delay: 0.2s !important; }}
    .mlops-dots span:nth-child(3) {{ animation-delay: 0.4s !important; }}
  `;
  parent.head.appendChild(style);

  // Chat bubble button
  const bubble = parent.createElement('button');
  bubble.id        = 'mlops-chat-bubble';
  bubble.innerHTML = '<svg viewBox="0 0 24 24" width="24" height="24" fill="white"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z"/></svg>';
  bubble.title     = 'Ask about your model';
  parent.body.appendChild(bubble);

  // Chat panel
  const panel   = parent.createElement('div');
  panel.id       = 'mlops-chat-panel';
  panel.innerHTML = `
    <div id="mlops-header">
      <div>
        <div>Model Assistant</div>
        <div class="sub">Llama 3.2 · RAG · Local</div>
      </div>
      <button id="mlops-close">✕</button>
    </div>
    <div id="mlops-messages">
      <div class="mlops-msg bot">Hi! Ask me about model performance, drift, or features.</div>
    </div>
    <div id="mlops-quick">
      <button class="mlops-qbtn">Best model?</button>
      <button class="mlops-qbtn">Retrain now?</button>
      <button class="mlops-qbtn">Drift summary</button>
    </div>
    <div id="mlops-input-row">
      <input id="mlops-input" type="text" placeholder="Ask about drift, models..." />
      <button id="mlops-send">➤</button>
    </div>
  `;
  parent.body.appendChild(panel);

  // Toggle open/close
  bubble.addEventListener('click', () => panel.classList.toggle('mlops-open'));
  parent.getElementById('mlops-close').addEventListener('click', () => panel.classList.remove('mlops-open'));

  // Quick buttons
  parent.querySelectorAll('.mlops-qbtn').forEach(btn => {{
    btn.addEventListener('click', () => sendMsg(btn.textContent));
  }});

  // Send on Enter
  parent.getElementById('mlops-input').addEventListener('keydown', e => {{
    if (e.key === 'Enter') sendMsg();
  }});

  // Send button
  parent.getElementById('mlops-send').addEventListener('click', () => sendMsg());

  function addMsg(text, role) {{
    const msgs = parent.getElementById('mlops-messages');
    const div  = parent.createElement('div');
    div.className   = 'mlops-msg ' + role;
    div.textContent = text;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    return div;
  }}

  function addTyping() {{
    const msgs = parent.getElementById('mlops-messages');
    const div  = parent.createElement('div');
    div.id        = 'mlops-typing';
    div.className = 'mlops-msg typing';
    div.innerHTML = '<div class="mlops-dots"><span></span><span></span><span></span></div>';
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
  }}

  function removeTyping() {{
    const t = parent.getElementById('mlops-typing');
    if (t) t.remove();
  }}

  async function sendMsg(preset) {{
    const input   = parent.getElementById('mlops-input');
    const sendBtn = parent.getElementById('mlops-send');
    const q       = preset || input.value.trim();
    if (!q) return;

    addMsg(q, 'user');
    input.value      = '';
    sendBtn.disabled = true;
    addTyping();

    const prompt = 'You are an assistant ONLY for this MLOps retail demand forecasting platform. Answer ONLY questions about model performance, data drift, features, and retraining decisions for this project. If the question is unrelated to this platform, respond with: I can only answer questions about this MLOps platform. Use ONLY the facts in the context below. Do not use outside knowledge.\\n\\nContext:\\n' +
      RAG_CONTEXT + '\\n\\nQuestion: ' + q +
      '\\nAnswer concisely using only facts from the context:';

    try {{
      const res = await fetch(OLLAMA_URL, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          model: MODEL, prompt: prompt, stream: false,
          options: {{ temperature: 0.1, num_predict: 200 }}
        }})
      }});
      removeTyping();
      if (!res.ok) {{
        addMsg('Error reaching Ollama. Is it running with CORS enabled?', 'bot');
      }} else {{
        const data = await res.json();
        addMsg(data.response || 'No response.', 'bot');
      }}
    }} catch(err) {{
      removeTyping();
      addMsg('Cannot reach Ollama. Run: OLLAMA_ORIGINS=\\'*\\' ollama serve', 'bot');
    }}
    sendBtn.disabled = false;
    input.focus();
  }}
}})();
</script>
"""
    # height=1 so the iframe exists but is invisible
    # The script runs and injects into the parent page
    components.html(html, height=1, scrolling=False)
