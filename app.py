
# ============================================================
# ScamShield Bengaluru — explainable digital safety assistant
# ============================================================

# Imports
import re
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Config: must be the first Streamlit command
st.set_page_config(page_title="ScamShield Bengaluru", page_icon="🛡️", layout="wide")

# Visual design system: navy, teal, accessible typography, responsive layout
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#0b1120;--panel:#121c2e;--panel2:#18263a;--border:#293a50;--teal:#40e0c0;--text:#f1f5fb;--muted:#a7b6c9}
.stApp{background:radial-gradient(ellipse at 15% 0%,#17304a 0%,transparent 36%),radial-gradient(ellipse at 95% 35%,#10352f 0%,transparent 27%),var(--bg);color:var(--text);font-family:'DM Sans',sans-serif}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#101b2b,#0b1321);border-right:1px solid var(--border)}
h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:-.5px}
h1{font-size:clamp(2rem,4vw,3rem)!important;font-weight:800!important;line-height:1.15!important}
h2{font-size:1.55rem!important;font-weight:700!important}h3{font-size:1.12rem!important;font-weight:700!important}
p,label,li{line-height:1.65}[data-testid="stCaptionContainer"]{color:var(--muted)}
[data-testid="stTabs"] [role="tablist"]{gap:8px;border-bottom:1px solid var(--border)}
[data-testid="stTabs"] button[role="tab"]{border-radius:10px 10px 0 0;padding:12px 16px;font-weight:600}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{color:var(--teal);border-bottom:2px solid var(--teal)}
[data-testid="stTextArea"] textarea{background:#101a2a!important;color:var(--text)!important;border:1px solid #34485f!important;border-radius:14px!important;padding:16px!important;font-size:1rem!important;line-height:1.7!important;min-height:160px}
[data-testid="stTextArea"] textarea:focus{border-color:var(--teal)!important;box-shadow:0 0 0 2px #40e0c025!important}
[data-testid="stSelectbox"]>div>div{background:#121e30;border-radius:10px;border-color:var(--border)}
.stButton>button{min-height:46px;border-radius:11px;padding:.65rem 1.1rem;font-weight:700;border:1px solid #34485f;background:#17263a;color:var(--text);transition:all 160ms ease}
.stButton>button:hover{border-color:var(--teal);color:var(--teal);transform:translateY(-1px)}
.stButton>button[kind="primary"]{background:linear-gradient(115deg,#16a995,#40e0c0);color:#061b20;border:none;box-shadow:0 5px 20px #40e0c020}
.stButton>button[kind="primary"]:hover{background:#6aecd4;color:#061b20}
[data-testid="stMetric"]{background:linear-gradient(145deg,#17263a,#111b2b);border:1px solid var(--border);border-radius:15px;padding:18px 20px;box-shadow:0 8px 24px #00000015}
[data-testid="stMetricLabel"]{color:var(--muted);font-weight:600}[data-testid="stMetricValue"]{font-family:'Manrope',sans-serif;font-weight:800}
[data-testid="stAlert"]{border-radius:12px}[data-testid="stExpander"]{border:1px solid var(--border);border-radius:12px;background:#111b2b}
[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:12px;overflow:hidden}
hr{border-color:var(--border);margin:1.5rem 0}
@media(max-width:768px){.block-container{padding:1.2rem 1rem}h1{font-size:2rem!important}[data-testid="stTabs"] button[role="tab"]{padding:10px 7px;font-size:.82rem}}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Model loading: lazy-loaded only when an SLM action is pressed
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_slm(model_name):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    return tokenizer, model


def slm_chat(system_prompt: str, user_prompt: str, max_new_tokens: int = 256) -> str:
    import torch
    tokenizer, model = load_slm(st.session_state.model_name)
    user_prompt = user_prompt[:6000]
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)
    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ------------------------------------------------------------
# Helpers: JSON validation, safe types, and prompt transparency
# ------------------------------------------------------------
def parse_json(text):
    try:
        text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group(0)) if match else None
    except Exception:
        return None


def slm_json(system_prompt, user_prompt, max_new_tokens=300):
    for attempt in range(2):
        try:
            prompt = user_prompt + ("\nReturn ONLY valid JSON. No extra text." if attempt else "")
            result = parse_json(slm_chat(system_prompt, prompt, max_new_tokens))
            if isinstance(result, dict):
                return result
        except Exception:
            if attempt == 1:
                return None
    return None


def safe_float(value, default=0.0):
    try:
        number = float(value)
        return number if number == number and abs(number) != float("inf") else default
    except (TypeError, ValueError, OverflowError):
        return default


def safe_list(value, default=None):
    if default is None:
        default = []
    if isinstance(value, list):
        return [str(item)[:300] for item in value[:20]]
    if isinstance(value, str) and value.strip():
        return [value[:300]]
    return default.copy()


def show_prompt(system_prompt, user_prompt):
    with st.expander("See what we asked the SLM"):
        st.markdown("**System prompt**")
        st.code(system_prompt, language=None)
        st.markdown("**User prompt**")
        st.code(user_prompt, language=None)


def safe_actions(items):
    actions = safe_list(items)
    if actions:
        return actions[:3]
    return ["Do not click links or share OTPs.", "Verify using an official contact method obtained independently.", "Ask a trusted family member for help if unsure."]

# ------------------------------------------------------------
# Built-in sample data: 18 scam and 12 safe illustrative messages
# ------------------------------------------------------------
SCAM = [
    "Dear customer your BESCOM power will be disconnected tonight 9.30pm call 9876501001",
    "Sir nimma KYC block aagide update maadi link click maadi",
    "Your bank account will be frozen today. Share your OTP immediately.",
    "Your FedEx parcel has illegal items. Mumbai police will call you.",
    "You have won a lottery prize of Rs 250000. Pay processing fee now.",
    "Earn guaranteed 30 percent weekly returns. Invest today through this link.",
    "Your son is under digital arrest. Transfer money immediately and tell nobody.",
    "KYC expired. Update your Aadhaar and bank details at http://fake.example",
    "Electricity bill overdue. Pay Rs 1500 now or power will be disconnected.",
    "Customs has seized your parcel. Pay Rs 3000 to release it.",
    "Congratulations, claim your free gift. Enter your bank PIN to continue.",
    "Your SIM card will be blocked in two hours. Call 9876501002 now.",
    "Police case registered against you. Do not tell anyone. Transfer money now.",
    "Sir nimma account block aagutte. OTP heli, naavu help maadthivi.",
    "Work from home job: pay Rs 5000 registration fee for guaranteed salary.",
    "Click http://fraud.example and enter your UPI PIN to receive a refund.",
    "Your pension verification failed. Send Aadhaar and OTP immediately.",
    "Pay Rs 999 customs clearance fee or your parcel will be destroyed."
]
SAFE = [
    "Water supply will be unavailable from 10 AM to 12 PM tomorrow.",
    "Namaskara everyone, apartment meeting is at 6 PM in the clubhouse.",
    "Ganesha festival celebration starts at 5 PM on Saturday.",
    "A brown dog was seen near Puttenahalli lake. Please contact the RWA.",
    "Lift maintenance is scheduled for Sunday morning.",
    "Please keep your parking area clean. Thank you, RWA team.",
    "The community yoga class begins at 7 AM tomorrow.",
    "Milk delivery will be delayed by 15 minutes today.",
    "Happy birthday to our neighbour Ravi. Have a wonderful day!",
    "The gardener will trim the plants near Block B this afternoon.",
    "Residents can collect their festival passes from the office.",
    "Please submit maintenance suggestions before the monthly meeting."
]
EXAMPLES = [(message, "scam") for message in SCAM] + [(message, "safe") for message in SAFE]
SCAM_TYPES = ["Fake KYC", "Electricity bill", "Lottery/prize", "Job/investment", "Digital arrest", "Courier/customs", "OTP theft", "Not a scam"]
SAMPLE_MESSAGES = {
    "Choose an example...": "",
    "Fake BESCOM disconnection": SCAM[0],
    "Kanglish KYC warning": SCAM[1],
    "Courier customs threat": SCAM[3],
    "Lottery prize request": SCAM[4],
    "Normal RWA announcement": SAFE[1],
    "Water supply notice": SAFE[0]
}

# ------------------------------------------------------------
# NLP functions: baseline model and red-flag extraction
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_classic_model():
    texts = [message for message, label in EXAMPLES]
    labels = [label for message, label in EXAMPLES]
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(texts, labels)
    return pipeline


def classic_predict(message):
    model = train_classic_model()
    probabilities = model.predict_proba([message])[0]
    classes = list(model.named_steps["clf"].classes_)
    probability = float(probabilities[classes.index("scam")]) * 100
    return {"prediction": classes[probabilities.argmax()], "probability": round(probability, 1)}


def extract_red_flags(message):
    patterns = {
        "URLs": r"(?:https?://|www\.)[^\s<>]+",
        "Phone numbers": r"(?<!\w)(?:\+?91[\s-]?)?[6-9]\d(?:[\s-]?\d){8}(?!\w)",
        "UPI IDs": r"(?i)\b[a-z0-9._-]{2,}@[a-z][a-z0-9]{1,}",
        "Rupee amounts": r"(?i)(?:₹|rs\.?\s*)\s?\d[\d,]*(?:\.\d{1,2})?"
    }
    found = {}
    for label, pattern in patterns.items():
        matches = re.findall(pattern, message)
        if matches:
            found[label] = list(dict.fromkeys(matches))[:10]
    signals = [
        r"\b(?:urgent|immediately|right now|within \d+ hours)\b",
        r"\b(?:otp|one.time password|upi pin|password)\b",
        r"\b(?:disconnected|blocked|frozen|arrest|police case)\b",
        r"\b(?:guaranteed returns|processing fee|claim your prize)\b",
        r"\b(?:tell nobody|do not tell anyone|digital arrest)\b",
        r"\b(?:kyc|customs clearance|electricity bill)\b"
    ]
    for pattern in signals:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            found.setdefault("Suspicious language", []).append(match.group(0))
    return found

# Exact prompts from the problem specification
ANALYSIS_SYSTEM = ('You are a cyber-safety assistant for senior citizens in Bengaluru. Analyse the WhatsApp message. Respond ONLY with JSON: {"is_scam": true/false, "risk_score": 0-100, "scam_type": one of ["Fake KYC","Electricity bill","Lottery/prize","Job/investment","Digital arrest","Courier/customs","OTP theft","Not a scam"], "red_flags": [list of short phrases], "explanation": "2 simple sentences", "what_to_do": ["3 short safe actions"]}')
PARENTS_SYSTEM = "Rewrite this scam explanation for an elderly person in 3 very short, kind sentences. No technical words. Plain text only."


def normalize_analysis(data):
    if not isinstance(data, dict):
        return None
    is_scam = data.get("is_scam")
    if not isinstance(is_scam, bool):
        return None
    score = max(0, min(100, round(safe_float(data.get("risk_score"), 50))))
    category = str(data.get("scam_type", "Not a scam"))
    if category not in SCAM_TYPES:
        category = "Not a scam"
    conflict = ((not is_scam and score >= 71) or (is_scam and score <= 30) or (not is_scam and category != "Not a scam") or (is_scam and category == "Not a scam"))
    if conflict:
        category = "Needs verification"
    return {
        "is_scam": is_scam, "risk_score": score, "scam_type": category,
        "conflict": conflict, "red_flags": safe_list(data.get("red_flags")),
        "explanation": str(data.get("explanation", "This result needs further verification."))[:1000],
        "what_to_do": safe_actions(data.get("what_to_do"))
    }

# ------------------------------------------------------------
# UI: session state, branded header, sidebar, tabs and results
# ------------------------------------------------------------
for key, default in [("history", []), ("classic_result", None), ("slm_result", None), ("parents_explanation", None), ("message_input", "")]:
    if key not in st.session_state:
        st.session_state[key] = default


def choose_example():
    st.session_state.message_input = SAMPLE_MESSAGES.get(st.session_state.example_select, "")

st.markdown("""
<div style="padding:26px 28px;border-radius:20px;background:linear-gradient(120deg,#152c42 0%,#12372f 100%);border:1px solid #315c61;margin-bottom:24px;">
  <div style="color:#40e0c0;font-size:12px;font-weight:800;letter-spacing:2px;margin-bottom:12px;">DIGITAL SAFETY · BENGALURU</div>
  <div style="font-family:Manrope,sans-serif;font-size:clamp(27px,4vw,40px);font-weight:800;line-height:1.2;color:#f4f8ff;">🛡️ Stay alert. Stay protected.</div>
  <div style="color:#c0d0df;font-size:16px;margin-top:12px;max-width:760px;line-height:1.7;">Check suspicious WhatsApp messages, understand the warning signs, and learn the safest next step — in simple language.</div>
  <div style="margin-top:18px;color:#8cebd9;font-size:12px;font-weight:600;">AI-ASSISTED CHECKS &nbsp; · &nbsp; YOUR SAFETY COMES FIRST</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.selectbox("Small Language Model", ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"], index=0, key="model_name")
    st.markdown("### 🧠 Two ways to check")
    st.info("**Classic NLP** looks for learned word patterns. **SLM** considers context and generates explanations. Neither can guarantee a message is safe.")
    st.caption("Privacy reminder: do not enter real OTPs, passwords, bank details or other sensitive information in this demo.")

check_tab, history_tab, how_tab = st.tabs(["🔎 Check a message", "📊 History & stats", "💡 How it works"])

with check_tab:
    st.markdown("## Check a forwarded message")
    st.write("Paste the message below or start with a sample. Your results will appear underneath.")
    st.selectbox("Try a built-in example", list(SAMPLE_MESSAGES.keys()), key="example_select", on_change=choose_example)
    message = st.text_area("Message to check", key="message_input", height=160, placeholder="Paste a WhatsApp message here…")
    btn1, btn2 = st.columns(2)
    with btn1:
        classic_button = st.button("⚡ Check with Classic NLP", key="classic_button", type="primary")
    with btn2:
        slm_button = st.button("🛡️ Analyse with SLM", key="slm_button")

    if classic_button:
        if not message.strip():
            st.warning("Please paste a message before checking it.")
        else:
            try:
                with st.spinner("Checking message patterns…"):
                    result = classic_predict(message)
                    flags = extract_red_flags(message)
                st.session_state.classic_result = {"message": message, **result, "flags": flags}
                st.session_state.history.append({"message": message, "verdict": result["prediction"], "score": result["probability"], "type": "Classic NLP", "red_flags": flags})
                st.session_state.parents_explanation = None
            except Exception as exc:
                st.error("Classic analysis could not be completed. Please try again.")
                print("Classic NLP error:", repr(exc))

    if slm_button:
        if not message.strip():
            st.warning("Please paste a message before analysis.")
        else:
            try:
                with st.spinner("Thinking with the SLM… First run downloads the model."):
                    raw = slm_json(ANALYSIS_SYSTEM, "Analyse this message:\n" + message, max_new_tokens=300)
                result = normalize_analysis(raw)
                if result is None:
                    st.info("The SLM answer could not be read. Retry or select the 1.5B model. Classic NLP remains available.")
                else:
                    st.session_state.slm_result = {"message": message, **result}
                    st.session_state.history.append({"message": message, "verdict": "scam" if result["is_scam"] else "safe", "score": result["risk_score"], "type": result["scam_type"], "red_flags": result["red_flags"]})
                    st.session_state.parents_explanation = None
            except Exception as exc:
                st.error("The SLM could not complete the analysis. Try Classic NLP or select the smaller model.")
                print("SLM error:", repr(exc))

    classic = st.session_state.classic_result
    slm = st.session_state.slm_result
    if classic or slm:
        st.divider()
        st.markdown("## Your analysis")
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("### ⚡ Classic NLP")
            if classic:
                st.metric("Estimated scam probability", f'{classic["probability"]:.1f}%')
                st.markdown(f'**Predicted label:** `{classic["prediction"].upper()}`')
                st.caption("This is a model probability from a tiny illustrative dataset, not a verified real-world fraud rate.")
                st.markdown("#### 🚩 Detected red flags")
                if classic["flags"]:
                    for kind, values in classic["flags"].items():
                        st.markdown(f"**{kind}**: " + ", ".join(values))
                else:
                    st.write("No supported red-flag patterns detected. This does not prove the message is safe.")
            else:
                st.info("Run Classic NLP to see its result.")
        with right:
            st.markdown("### 🧠 SLM assessment")
            if slm:
                score = slm["risk_score"]
                level = "LOW" if score <= 30 else ("MODERATE" if score <= 70 else "HIGH")
                st.metric("Estimated risk score", f"{score}/100", delta=level, delta_color="off")
                st.progress(score / 100, text=f"Risk indicator · {level}")
                st.markdown(f'**Category:** `{slm["scam_type"]}`')
                if slm.get("conflict", False):
                    st.warning("The model returned conflicting signals. Do not rely on its verdict or score alone. Verify through an independently obtained official contact.")
                st.markdown("#### 💬 Why this may be risky")
                st.write(slm["explanation"])
                st.markdown("#### 🚩 Red flags reported by the SLM")
                if slm["red_flags"]:
                    for flag in slm["red_flags"]:
                        st.markdown(f"- {flag}")
                else:
                    st.write("The SLM reported no specific red flags.")
                st.markdown("#### ✅ What you should do next")
                for index, action in enumerate(slm["what_to_do"], 1):
                    st.markdown(f"**{index}.** {action}")
                st.caption("AI-generated assessment, not a verified fraud finding. A low score does not establish safety.")
                show_prompt(ANALYSIS_SYSTEM, "Analyse this message:\n" + slm["message"])
            else:
                st.info("Run SLM analysis to see its result.")

        if classic and slm and classic["message"] == slm["message"]:
            classic_scam = classic["prediction"] == "scam"
            if classic_scam != slm["is_scam"]:
                st.warning("The two models disagree. Pause and verify the message through an official contact method you find independently.")

    if slm:
        if st.button("💚 Explain to my parents", key="parents_button"):
            try:
                with st.spinner("Simplifying the explanation… First run downloads the model if not already loaded."):
                    simplified = slm_chat(PARENTS_SYSTEM, "Explanation: " + slm["explanation"] + "\nSafe actions: " + "; ".join(slm["what_to_do"]), max_new_tokens=100)
                st.session_state.parents_explanation = simplified
            except Exception as exc:
                st.error("Could not simplify the explanation. Please use the safety instructions above.")
                print("Parent explanation error:", repr(exc))
        if st.session_state.parents_explanation:
            st.markdown("### 💚 A simpler explanation")
            st.success(st.session_state.parents_explanation)
            show_prompt(PARENTS_SYSTEM, "Explanation: " + slm["explanation"] + "\nSafe actions: " + "; ".join(slm["what_to_do"]))

with history_tab:
    st.markdown("## Your checking history")
    st.caption("History is stored in this active Streamlit session only; it is not permanent.")
    history = st.session_state.history
    if history:
        rows = []
        for item in history:
            rows.append({"Verdict": item["verdict"].upper(), "Score": item["score"], "Analysis type": item["type"], "Message preview": item["message"][:90]})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        counts = pd.DataFrame({"Category": [item["type"] if item["type"] != "Classic NLP" else item["verdict"].title() for item in history]}).value_counts().reset_index(name="Count")
        fig = px.pie(counts, names="Category", values="Count", hole=0.48, title="Analysis categories in this session", color_discrete_sequence=["#40e0c0", "#4d8dff", "#ffb454", "#ed6a78", "#a78bfa"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#eaf1fb", legend_title_text="", margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig)
        if st.button("🗑️ Clear session history", key="clear_history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No messages checked yet. Analyse a sample message to begin.")

with how_tab:
    st.markdown("## How ScamShield works")
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("### ⚡ Classic NLP")
        st.write("TF-IDF converts words and short phrases into numeric features. Logistic Regression learns patterns from labelled examples. It is fast, but may miss unfamiliar wording and Kannada transliterations.")
        st.caption("Best for: fast baseline comparisons and visible word-pattern signals.")
    with col_b:
        st.markdown("### 🧠 Small language model")
        st.write("The selected Qwen model analyses the message using context and instructions. It can generate explanations and suggested actions, but can still hallucinate or misclassify messages.")
        st.caption("Best for: contextual analysis and plain-language explanations.")
    st.divider()
    st.markdown("### 🧭 Understanding the risk score")
    st.write("Low: 0–30 · Moderate: 31–70 · High: 71–100. These are demonstration thresholds, not validated boundaries. The SLM score is an estimate, not a calibrated probability.")
    st.markdown("### 🔐 Stay safe")
    st.write("Never share OTPs, UPI PINs or passwords. Do not transfer money because of a threatening message. Verify requests using contact details obtained independently. If money has already been lost in India, contact your bank immediately and report it through the official 1930 cybercrime helpline or cybercrime.gov.in.")
    st.warning("This prototype uses a very small illustrative dataset. Do not use it as the sole basis for financial or safety decisions. Real deployment requires representative multilingual data, independent evaluation, access controls and privacy safeguards.")
