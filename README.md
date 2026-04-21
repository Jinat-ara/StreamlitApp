# Cognitive Accessibility Analyzer for AI-generated Images

A public Streamlit app for assessing AI-generated images using lightweight cognitive accessibility proxies such as clutter, clarity, contrast, brightness, focus, and OCR-based text quality.

## Files

- `app.py` - the Streamlit app
- `requirements.txt` - Python dependencies
- `packages.txt` - Linux system package needed for OCR on Streamlit Community Cloud

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy free on Streamlit Community Cloud

1. Create a new public GitHub repository.
2. Upload these files to the repository root:
   - `app.py`
   - `requirements.txt`
   - `packages.txt`
   - `README.md`
3. Go to Streamlit Community Cloud.
4. Sign in with GitHub.
5. Click **Create app**.
6. Choose your repository, branch, and set the main file path to `app.py`.
7. Deploy.

## OCR note

This version removes the Windows-only hardcoded Tesseract path. It now:

- uses `TESSERACT_CMD` if you set it manually
- otherwise auto-detects `tesseract` on the host
- otherwise disables OCR gracefully

That makes it suitable for both Windows and Linux hosting.

## Optional improvements before publishing

- pin dependency versions after your first successful deployment
- add a sample screenshot or GIF to the README
- add a license file if you want others to reuse the code
- add a short disclaimer that this is a proxy-based analyzer, not a clinical accessibility assessment tool
