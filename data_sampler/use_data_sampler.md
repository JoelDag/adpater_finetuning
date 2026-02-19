# FineWeb2 Subset Downloader

## Usage

1. Install required packages (`pip install datatrove`).
3. Run the script:
   ```bash
   python sample_fineweb.py
--- 

- Languages are selected from `language_codes.py`.
- `limit` specifies how many documents to download per language.
  - For the top 20 languages, `limit=500000` results in ~30 GB downloaded data.
  - For smaller languages, sizes vary greatly.

- `tasks=3` controls how many processes download in parallel **per language**.
  - High `tasks` values may cause **429 Too Many Requests** errors for low-resource languages.

Try smaller limit values and a few languages first to validate the pipeline.
