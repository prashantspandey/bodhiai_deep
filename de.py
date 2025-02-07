import streamlit as st
import requests
import json
import time
import re

# Azure configuration
AZURE_ENDPOINT = "https://ai-aihub2573706963054.services.ai.azure.com/models/chat/completions"
API_KEY = "4ZKiVgYHfBBHIHijqHMVtE6xh5ABLfFslHtElxGLuMZwRL839BI2JQQJ99BBACYeBjFXJ3w3AAAAACOGpNZC"
DEPLOYMENT_NAME = "DeepSeek-R1"

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'total_tokens' not in st.session_state:
        st.session_state.total_tokens = 0
    if 'debug_log' not in st.session_state:
        st.session_state.debug_log = []
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = []
    if 'current_chat_index' not in st.session_state:
        st.session_state.current_chat_index = -1

def log_debug(message, level="info"):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.debug_log.append({"time": timestamp, "message": message, "level": level})

def update_token_count(content):
    st.session_state.total_tokens += len(content.split()) + len([c for c in content if c in '.,!?;:'])

def style_thinking(text):
    pattern = r"<think>(.*?)</think>"
    def repl(match):
        inner = match.group(1).strip()
        return f'<div class="thinking-box">💭 {inner}</div>'
    return re.sub(pattern, repl, text, flags=re.DOTALL)

def process_stream(response, message_placeholder):
    buffer = ""
    display_text = ""

    for line in response.iter_lines():
        if not line:
            continue
           
        try:
            line = line.decode('utf-8').strip()
            if line.startswith("data: "):
                line = line[6:]
            if not line:
                continue
               
            json_response = json.loads(line)
            choices = json_response.get('choices', [])
            if not choices:
                continue
               
            delta = choices[0].get('delta', {})
            if 'content' not in delta:
                continue
               
            content = delta.get('content', '')
            if content:
                buffer += content
                update_token_count(content)
               
                if len(buffer) > 20 or any(buffer.endswith(x) for x in ['.', '!', '?', '\n']):
                    for char in buffer:
                        display_text += char
                        message_placeholder.markdown(style_thinking(display_text) + "▌", unsafe_allow_html=True)
                        time.sleep(0.02)
                    buffer = ""
                   
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            continue

    if buffer:
        for char in buffer:
            display_text += char
            message_placeholder.markdown(style_thinking(display_text) + "▌", unsafe_allow_html=True)
            time.sleep(0.02)

    if display_text:
        message_placeholder.markdown(style_thinking(display_text), unsafe_allow_html=True)
        return display_text
    return None

def get_streaming_response(messages, max_retries=3):
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
        "x-ms-model-mesh-model-name": DEPLOYMENT_NAME
    }
   
    payload = {
        "messages": messages,
        "max_tokens": 4000,
        "temperature": 0.7,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": True
    }

    retries = 0
    while retries < max_retries:
        try:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
               
                with requests.post(
                    f"{AZURE_ENDPOINT}?api-version=2024-05-01-preview",
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=120
                ) as response:
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 60))
                        time.sleep(retry_after)
                        retries += 1
                        continue
                       
                    response.raise_for_status()
                    return process_stream(response, message_placeholder)
                   
        except requests.exceptions.Timeout:
            if retries < max_retries - 1:
                time.sleep(2 ** retries)
                retries += 1
                continue
            st.error("Request timed out. Please try again.")
            return None
           
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None
           
    st.error("Max retries reached. Please try again later.")
    return None

def save_current_chat():
    if st.session_state.messages:
        title = st.session_state.messages[0]["content"] if st.session_state.messages[0]["role"] == "user" else f"Chat {len(st.session_state.chat_sessions) + 1}"
        new_chat = {"title": title, "messages": st.session_state.messages.copy()}
        st.session_state.chat_sessions.append(new_chat)
        st.session_state.current_chat_index = len(st.session_state.chat_sessions) - 1

def load_chat(index):
    st.session_state.current_chat_index = index
    st.session_state.messages = st.session_state.chat_sessions[index]["messages"].copy()
    st.session_state.total_tokens = sum(
        len(msg["content"].split()) + len([c for c in msg["content"] if c in '.,!?;:'])
        for msg in st.session_state.messages
    )

def main():
    st.set_page_config(page_title="Bodhi AI Chat", layout="wide")
    init_session_state()

    # Add MathJax JavaScript
    st.markdown("""
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
        <script>
        MathJax.Hub.Config({
            tex2jax: {
                inlineMath: [['$','$'], ['\\(','\\)']],
                displayMath: [['$$','$$'], ['\\[','\\]']],
                processEscapes: true,
                processEnvironments: true
            },
            displayAlign: 'center',
            "HTML-CSS": {
                styles: {'.MathJax_Display': {"margin": 0}},
                linebreaks: { automatic: true }
            }
        });
        </script>
    """, unsafe_allow_html=True)

    # Sidebar chat selection
    with st.sidebar:
        st.title("📝 Bodhi AI Chat")
        st.header("Chats")
        chat_options = ["New Chat"] + [chat["title"] for chat in st.session_state.chat_sessions]
        selected_chat = st.radio("Select a chat", options=chat_options, index=0)
        
        if selected_chat != "New Chat":
            for i, chat in enumerate(st.session_state.chat_sessions):
                if chat["title"] == selected_chat:
                    load_chat(i)
                    break
        
        if st.button("New Chat"):
            if st.session_state.messages:
                save_current_chat()
            st.session_state.messages = []
            st.session_state.total_tokens = 0
            st.session_state.current_chat_index = -1
            st.experimental_rerun()

    # Custom styling
    st.markdown("""
        <style>
        .stApp { background-color: #1E1E1E; color: white; }
        .token-counter { position: fixed; top: 10px; right: 10px; background-color: #2D2D2D; 
                        padding: 5px 10px; border-radius: 5px; }
        .thinking-box { background-color: #2D2D2D; border-left: 4px solid #0078D4;
                       padding: 10px; margin: 10px 0; font-style: italic; color: #B0B0B0; }
        .stChatInput { position: relative; }
        .youtube-icon {
            position: absolute;
            left: 15px;
            top: 50%;
            transform: translateY(-50%);
            z-index: 2;
            color: #ff0000;
        }
        .stChatInput textarea {
            padding-left: 40px !important;
            background: transparent !important;
            border: 1px solid #404040 !important;
        }
        .MathJax_Display {
            overflow-x: auto;
            overflow-y: hidden;
            padding: 0.5em 0;
        }
        .MathJax {
            color: #ffffff !important;
        }
        
        /* Welcome message styles */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        .welcome-container {
            display: flex;
            align-items: center;
            gap: 20px;
            margin: 50px 0;
            opacity: 1;
            transition: opacity 0.5s ease;
        }
        
        .welcome-circle {
            width: 50px;
            height: 50px;
            background: linear-gradient(45deg, #0078D4, #00A5FF);
            border-radius: 50%;
            animation: float 3s ease-in-out infinite;
        }
        
        .welcome-text {
            font-size: 24px;
            color: white;
        }
        
        .welcome-subtext {
            font-size: 18px;
            color: #B0B0B0;
            margin-top: 10px;
        }
        
        /* Hide welcome message when chat has messages */
        .welcome-hidden {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display token counter
    st.markdown(f'<div class="token-counter">🔤 Tokens: {st.session_state.total_tokens}</div>', 
               unsafe_allow_html=True)
    
    # Display welcome message if no messages exist
    if not st.session_state.messages:
        st.markdown("""
            <div class="welcome-container">
                <div class="welcome-circle"></div>
                <div>
                    <div class="welcome-text"><strong>Hello! This is Bodhi AI</strong></div>
                    <div class="welcome-subtext">How can I help you today and forever?</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            content = style_thinking(content)
            st.markdown(content, unsafe_allow_html=True)
            st.markdown("""
                <script>
                if (typeof MathJax !== 'undefined') {
                    MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
                }
                </script>
            """, unsafe_allow_html=True)

    # Chat input and response handling
    if prompt := st.chat_input("What would you like to know?"):
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt}) 
        messages = [
            {"role": "system", "content": "You are a helpful assistant. You can use LaTeX for mathematical equations by enclosing them in $ symbols for inline equations or $$ for display equations."},
            *st.session_state.messages
        ]
        
        response = get_streaming_response(messages)
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()