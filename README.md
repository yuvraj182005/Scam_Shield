# 🛡️ ScamShield Bengaluru

**Stay alert. Stay protected.** ScamShield Bengaluru is an AI-assisted digital safety application that helps people, especially senior citizens and apartment communities, assess suspicious messages commonly circulated through WhatsApp, SMS and other communication channels.

The application combines a lightweight machine-learning baseline with a locally loaded Small Language Model (SLM) to identify potential warning signs, explain why a message may be suspicious, and suggest safer next steps. It can also prepare a structured cybercrime report summary for the user to review before filing through India's official reporting channels.

> **Important:** ScamShield is an educational prototype, not an official government service or a verified fraud-detection system. Its scores and classifications are estimates and must not be treated as proof that a message is safe or fraudulent.

---

## ✨ Features

### 1. Message scam analysis
- Paste a forwarded message or select a built-in example.
- Analyse messages with a classic NLP model or a Small Language Model.
- View a scam probability estimate from the classic classifier.
- View an SLM-generated risk score, suspected scam category, explanation and recommended actions.
- Compare model outputs and receive a warning when their results conflict.

### 2. Red-flag detection
Uses local pattern matching to identify items such as:
- URLs and website links
- Phone numbers
- UPI IDs
- Rupee amounts
- Potentially suspicious wording, including urgency, OTP requests, account blocking and threats

These indicators are clues for review, not proof of fraud.

### 3. Explain it in simple language
The **Explain to my parents** feature asks the SLM to rewrite the assessment in short, simple sentences intended to be easier for older adults to understand.

### 4. History and statistics
- Keeps a history of analyses in the current Streamlit session.
- Displays message previews, verdicts, scores and analysis types.
- Shows a Plotly chart of analysis categories.
- Allows the user to clear the current session history.

### 5. Prepare a cybercrime report
- Creates a structured report summary using the message and available analysis.
- Collects incident details such as date received, communication channel, reported financial or information impact, optional amount lost and additional observations.
- Includes detected indicators, model assessment and explanation where available.
- Allows the user to review and download the report summary.
- Provides a route to India's official National Cyber Crime Reporting Portal.

**The app does not submit a complaint automatically.** The user must review the generated summary, open the official portal and complete the reporting process themselves. Do not include OTPs, passwords, UPI PINs or unnecessary sensitive personal information in the app or report.

If money has been lost, contact the bank or payment provider immediately and call India's cyber financial fraud helpline **1930** as soon as possible.

Official reporting portal: https://cybercrime.gov.in/

### 6. Privacy-conscious prototype
- No paid AI API keys are required.
- The SLM is loaded locally in the Colab runtime after the user requests an SLM action.
- Analysis history is held in Streamlit session state and is not designed as permanent storage.
- The app does not need users to upload a dataset to try the built-in examples.

The Colab-hosted app is exposed through a temporary public Cloudflare tunnel. Do not use real sensitive messages or personal information in this prototype.

---

## 🧠 How it works

```text
User pastes a message
        |
        v
Text input and local red-flag extraction
        |
        +----------------------------+
        |                            |
        v                            v
Classic NLP                   Small Language Model
TF-IDF + Logistic             Qwen2.5 Instruct
Regression                    (0.5B or 1.5B)
        |                            |
        v                            v
Scam probability              Risk estimate, category,
and predicted label           explanation and safe actions
        |                            |
        +-------------+--------------+
                      |
                      v
             Compare and flag conflicts
                      |
          +-----------+------------+
          |           |            |
          v           v            v
        Results     History     Report summary
                                  |
                                  v
                         User reviews and submits
                         through official portal
```

### Classic NLP
The baseline is a scikit-learn pipeline:

- `TfidfVectorizer(ngram_range=(1, 2), lowercase=True)`
- `LogisticRegression(max_iter=1000)`

It is trained on the small illustrative labelled examples included in the application. Its output is useful for demonstrating a traditional text-classification workflow, but it is not a calibrated real-world fraud probability.

### Small Language Model
The app offers two model choices:

- `Qwen/Qwen2.5-0.5B-Instruct` — default, smaller and generally faster.
- `Qwen/Qwen2.5-1.5B-Instruct` — larger option that may provide stronger responses but requires more compute and memory.

The model is downloaded from Hugging Face when first needed. It runs inside the Colab runtime; no paid inference API is used.

---

## 🧰 Technology stack

| Area | Technology |
|---|---|
| User interface | Streamlit |
| Language | Python |
| Data handling | pandas |
| Classic text classification | scikit-learn |
| Text features | TF-IDF |
| Classifier | Logistic Regression |
| Small Language Model | Qwen2.5 Instruct |
| Model runtime | PyTorch, Transformers, Accelerate |
| Charts | Plotly Express |
| Hosting for demo | Google Colab + Cloudflare Quick Tunnel |

---

## 🚀 Run in Google Colab

The project is designed to run as a **single Google Colab code cell**. The launcher installs the required packages, writes `app.py`, checks its syntax, starts Streamlit and opens a temporary Cloudflare tunnel.

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.
3. Copy the complete single-cell launcher code into one cell.
4. Optionally select **Runtime → Change runtime type → T4 GPU** for faster SLM inference, if available.
5. Run the cell and wait for the public `trycloudflare.com` URL.
6. Open the URL to use ScamShield Bengaluru.
7. Keep the Colab runtime running while using the application.

The first SLM analysis downloads the selected model from Hugging Face. This can take several minutes and requires an internet connection and sufficient runtime storage and memory. A GPU is optional, but can make inference faster.

### Packages used by the launcher

```bash
pip install -q streamlit "transformers>=4.43" accelerate sentence-transformers scikit-learn nltk plotly
```

PyTorch is supplied by the Colab runtime and is not reinstalled by the launcher.

---

## 🧪 Try the application

The application includes illustrative examples such as:

**Potential electricity-disconnection scam**
```text
Your BESCOM power will be disconnected tonight.
Pay immediately using this link.
```

**Potential Kanglish KYC scam**
```text
Sir nimma KYC block aagide, OTP heli account unblock maadthivi.
```

**Normal apartment announcement**
```text
Namaskara everyone, the apartment RWA meeting is at 6 PM in the clubhouse.
```

Expected classifications are not guaranteed. Model responses can vary, and the small sample dataset is not sufficient to establish reliable accuracy.

---

## 📁 Project structure

The Colab launcher creates the Streamlit application file at runtime:

```text
ScamShield-Bengaluru/
├── README.md
└── scamshield_bengaluru_colab.py
```

When the Colab cell runs, it writes and launches:

```text
app.py
streamlit.log
cloudflared.log
cloudflared
```

The generated runtime files are created in the Colab environment.

---

## ⚠️ Limitations and responsible use

- The built-in dataset contains only a small number of illustrative scam and safe messages. It is not representative of real-world scam prevalence.
- The classic classifier may overfit and may miss new scam patterns, Kannada script, Kanglish transliterations and unfamiliar wording.
- The SLM can hallucinate, misclassify messages or generate incorrect explanations and actions.
- A risk score is an estimate, not a verified probability or official fraud determination.
- A low score does not mean a message is safe. A high score does not independently prove criminal activity.
- Red-flag extraction uses patterns and does not verify whether a URL, phone number or UPI ID is malicious.
- The report generator prepares a draft only; it does not file, transmit or track a complaint.
- Streamlit session history is temporary and may disappear when the session or Colab runtime ends.
- Colab and its temporary public tunnel are intended for demonstrations, not dependable production hosting.

**Before real-world deployment:** build a larger, consented and independently labelled multilingual dataset; evaluate precision, recall, F1-score and false-negative rates; calibrate scores; test accessibility with senior citizens; and implement appropriate authentication, secure hosting, data retention controls and privacy safeguards.

---

## 🔐 Safety guidance

- Never share an OTP, password, UPI PIN or banking credentials.
- Do not click suspicious links or call phone numbers included in threatening messages.
- Verify unexpected requests using contact details obtained independently from an official source.
- If money has been transferred, contact your bank immediately and call **1930**.
- File complaints through the official portal: https://cybercrime.gov.in/

---

## 🤝 Contributing

Contributions that improve accessibility, multilingual coverage, dataset quality, explainability, testing and privacy are welcome.

Before contributing:
1. Describe the issue or improvement.
2. Avoid committing real victims' personal information or unredacted scam conversations.
3. Use synthetic or properly anonymised examples.
4. Include tests and explain any changes to model behaviour.

---

## 📄 License

Choose and add an open-source license before distributing or accepting external contributions. Until a license is added, reuse and redistribution permissions are not explicitly granted.

---

## 👨‍💻 Project

**ScamShield Bengaluru** — an AI-assisted digital safety prototype designed to help people recognise suspicious messages, understand potential warning signs and prepare more structured cybercrime reports.
