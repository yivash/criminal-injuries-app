# Criminal Injuries Compensation Scheme 2012 — RAG Demo

## Disclaimer

This free, independent educational demo shows how retrieval-augmented generation (RAG) can answer questions about the *Criminal Injuries Compensation Scheme 2012*. It is an unofficial project.

Scheme text and quoted passages are reproduced under the [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/). The original application code is available under the [MIT License](LICENSE).

## Data source

[Criminal Injuries Compensation Scheme 2012 (amended) — GOV.UK](https://www.gov.uk/government/publications/criminal-injuries-compensation-scheme-2012-amended--2)

## Setup and run locally

Run the following commands from the project root.

1. **Create and activate a virtual environment** (Windows PowerShell).

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. **Install dependencies**.
   ```powershell
   pip install -r requirements.txt
   ```
2. **Add the Scheme PDF.** Download the amended Scheme from the link above and save it as `data/criminal-injuries-compensation-scheme-2012.pdf`.

3. **Prepare the retrieval artefacts.**

   ```powershell
   python ./src/dataprep.py
   ```
   Inspect all required artifacts in the `indexes` folder.

4. **Run the Streamlit application.**

   ```powershell
   streamlit run ./src/app.py
   ```
