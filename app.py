"""
Streamlit frontend for the Python Test Case Generator.
-------
Run with:
    streamlit run app.py

Requires:
    pip install streamlit
"""

import streamlit as st
import time
from inference import generate_inputs

# Toggle this to False once the real model is ready
USE_MOCK = False


# Model loading (when model is ready)
@st.cache_resource
def load_model():
    """Load the trained model and tokenizer once and cache it."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    model_path = "./testgen_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model


def generate_tests_real(code: str) -> str:
    """Run inference using the trained model."""
    import torch
    from utils import (
        extract_signature,
        extract_func_name,
        extract_body,
        extract_param_names,
    )

    tokenizer, model = load_model()
    device = next(model.parameters()).device

    signature = extract_signature(code)
    func_name = extract_func_name(code)
    body = extract_body(code)
    valid_params = extract_param_names(code, func_name)

    # generate_inputs now returns full "assert candidate(...) == ..." strings
    assertions = generate_inputs(
        model=model,
        tokenizer=tokenizer,
        device=device,
        signature=signature,
        body=body,
        valid_params=valid_params,
    )

    if not assertions:
        return "No valid test assertions generated."

    return "\n".join(assertions)


# Mock model (used while real model is being trained)
def generate_tests_mock(code: str) -> str:
    """Return realistic-looking fake output for UI development."""
    time.sleep(1.5)  # simulate inference latency
    return """\
assert candidate(s="") == 0
assert candidate(s="a") == 1
assert candidate(s="abcabcbb") == 3
assert candidate(s="bbbbb") == 1
assert candidate(s="pwwkew") == 3
assert candidate(s="dvdf") == 3"""


# Route to mock or real
def generate_tests(code: str) -> str:
    if USE_MOCK:
        return generate_tests_mock(code)
    return generate_tests_real(code)


# Page config
st.set_page_config(
    page_title="PyTestGen",
    layout="wide",
)

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    /* Base */
    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    /* Dark background */
    .stApp {
        background-color: #0d0d0d;
        color: #f0f0f0;
    }

    /* Hide default streamlit header/footer */
    #MainMenu, footer, header { visibility: hidden; }

    /* Title */
    .title-block {
        padding: 2.5rem 0 1rem 0;
        border-bottom: 1px solid #2a2a2a;
        margin-bottom: 2rem;
    }
    .title-block h1 {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        color: #f0f0f0;
        letter-spacing: -1px;
        margin: 0;
    }
    .title-block h1 span {
        color: #00ff87;
    }
    .title-block p {
        color: #666;
        font-size: 1rem;
        margin-top: 0.4rem;
    }

    /* Section labels */
    .section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #00ff87;
        margin-bottom: 0.6rem;
    }

    /* Code areas */
    .stTextArea textarea {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        background-color: #111 !important;
        color: #e0e0e0 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 6px !important;
        caret-color: #00ff87;
    }
    .stTextArea textarea:focus {
        border-color: #00ff87 !important;
        box-shadow: 0 0 0 1px #00ff87 !important;
    }

    /* Output box */
    .output-box {
        background-color: #111;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        padding: 1.2rem 1.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #e0e0e0;
        min-height: 200px;
        white-space: pre-wrap;
        word-break: break-word;
        line-height: 1.7;
    }
    .output-box .kw-assert  { color: #00ff87; }
    .output-box .kw-eq      { color: #888; }
    .output-box .empty      { color: #444; font-style: italic; }

    /* Generate button */
    .stButton > button {
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 1px !important;
        background-color: #00ff87 !important;
        color: #0d0d0d !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        transition: opacity 0.15s ease !important;
    }
    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        padding: 2px 10px;
        border-radius: 20px;
        margin-left: 12px;
        vertical-align: middle;
    }
    .status-mock {
        background: #2a1f00;
        color: #ffaa00;
        border: 1px solid #ffaa00;
    }
    .status-real {
        background: #001a0d;
        color: #00ff87;
        border: 1px solid #00ff87;
    }

    /* Copy button */
    .copy-btn {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #444;
        cursor: pointer;
        float: right;
        margin-bottom: 0.4rem;
    }

    /* Divider */
    hr {
        border-color: #2a2a2a !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #00ff87 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
status_badge = (
    '<span class="status-badge status-mock">MOCK MODEL</span>'
    if USE_MOCK else
    '<span class="status-badge status-real">MODEL LOADED</span>'
)

st.markdown(f"""
<div class="title-block">
    <h1>Py<span>Test</span>Gen {status_badge}</h1>
    <p>Paste a Python function, get full test assertions back.</p>
</div>
""", unsafe_allow_html=True)

# Layout: two columns
col_in, col_out = st.columns(2, gap="large")

with col_in:
    st.markdown('<div class="section-label">Input — Python Function</div>', unsafe_allow_html=True)

    default_code = """\
def lengthOfLongestSubstring(s: str) -> int:
    d = {}
    res = 0
    left = 0
    for right, c in enumerate(s):
        if c in d and d[c] >= left:
            left = d[c] + 1
        d[c] = right
        res = max(res, right - left + 1)
    return res"""

    code_input = st.text_area(
        label="function_input",
        value=default_code,
        height=320,
        label_visibility="collapsed",
        placeholder="Paste your Python function here...",
    )

    generate_clicked = st.button("⚡ Generate Tests", use_container_width=True)

with col_out:
    st.markdown('<div class="section-label">Output — Generated Assertions</div>', unsafe_allow_html=True)

    # Output state
    if "output" not in st.session_state:
        st.session_state.output = None
    if "error" not in st.session_state:
        st.session_state.error = None

    if generate_clicked:
        if not code_input.strip():
            st.session_state.error = "Please paste a Python function first."
            st.session_state.output = None
        else:
            st.session_state.error = None
            with st.spinner("Generating..."):
                try:
                    result = generate_tests(code_input.strip())
                    st.session_state.output = result
                except Exception as e:
                    st.session_state.error = f"Model error: {e}"
                    st.session_state.output = None

    # Render output — highlight `assert` and `==` separately
    def highlight_assertion(line: str) -> str:
        """Color the assert keyword and == operator distinctly."""
        import re
        line = re.sub(
            r"^(assert)\s",
            r'<span class="kw-assert">\1</span> ',
            line,
        )
        line = re.sub(
            r"\s(==)\s",
            r' <span class="kw-eq">==</span> ',
            line,
        )
        return line

    if st.session_state.error:
        st.error(st.session_state.error)
    elif st.session_state.output:
        highlighted_lines = [
            highlight_assertion(line)
            for line in st.session_state.output.splitlines()
        ]
        highlighted = "\n".join(highlighted_lines)
        st.markdown(
            f'<div class="output-box">{highlighted}</div>',
            unsafe_allow_html=True
        )
        with st.expander("Copy the output..."):
            st.code(st.session_state.output, language="python")
    else:
        st.markdown(
            '<div class="output-box"><span class="empty">'
            'Generated test assertions will appear here...'
            '</span></div>',
            unsafe_allow_html=True
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<p style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #333; text-align: center;">
    ECS 189G · Bisrat · Dunn · Maram · Pham
</p>
""", unsafe_allow_html=True)