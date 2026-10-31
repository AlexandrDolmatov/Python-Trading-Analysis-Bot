# Validation and project boundaries

The existing README already described the strategy, data flow and limitations in useful detail. It was retained, with targeted additions for reproducible setup, local evidence and author links.

## Executed checks

- Python 3.12.14, with the 69 installed distributions pinned in `requirements.txt`; dependency compatibility check passed.
- `python examples/offline_check.py`: **6/6 passed**. Real indicator, persistence, detector, backtest and vectorbt code runs on explicitly synthetic input, without external requests. The positive signal fixture produces an executed backtest trade; this is not an historical result.
- `python -m unittest discover -s tests -v`: regression check for awaiting indicators in `run_selftest`; external calls are mocked.
- CLI help/import was checked during environment setup. No live monitoring loop, Telegram notification, historical download or parameter tuning was run in this portfolio review.
- GitHub Actions runs the same offline workflow; no remote CI outcome is claimed.

## Targeted corrections

The self-test previously passed an unawaited coroutine to the detector. It now awaits the asynchronous indicator function and propagates the self-test boolean as the CLI exit status. The aiohttp session now honors proxy environment settings, and SSL initialization uses an explicitly configured CA bundle when available, preserving certificate verification.

The real external workflow remains unverified despite these corrections. The self-test is still a diagnostic aid: some internal storage/error handling is permissive, so inspect its output and generated data rather than treating an exit code as proof of the entire system.

## Remaining limitations

- The application is a large single module with repeated helper definitions and mixed configuration, IO and strategy logic. No broad refactor was attempted.
- The source uses FMP `/api/v3` endpoints. Availability and intraday-history entitlement depend on the current FMP API/account; compatibility with a new subscription has not been verified.
- Telegram modes send messages; use only a chat you control. Credentials belong in local environment configuration, never in sample output.
- Autotuning writes the tracked `tickers_config.json`. Back it up and evaluate out of sample before retaining changes.
- Simulation assumptions and synthetic checks cannot establish profitability or predictive performance. No brokerage order execution is implemented.
- The bundled Windows TA-Lib wheel was retained to avoid deleting user files; the tested Linux setup does not use it.

## Credential review

A pattern scan of reachable text blobs found no matching GitHub/Telegram token, private key or literal API key in this repository. The code reads credentials from the environment and `.env`; the example file contains blank values. This limited scan is not an assurance that every possible secret format has been detected.

## Optional additional portfolio evidence

The checked-in offline output is reproducible and accurately labelled. For live results, provide a dated, anonymized run showing the command, data source, ticker/date window and output files, with API keys, chat IDs and personal messages removed. Do not publish performance claims without the underlying dataset and assumptions. A hosted always-on bot is unnecessary for this portfolio; a short authorized terminal recording is sufficient.
