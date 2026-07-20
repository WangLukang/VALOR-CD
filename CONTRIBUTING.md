# Contributing

Bug reports and focused pull requests are welcome.

1. Create a new branch from the default branch.
2. Install the development dependencies with `python -m pip install -e ".[dev]"`.
3. Run `python tools/check_release.py` and `pytest`.
4. Keep dataset paths relative and do not commit data, weights, outputs, credentials, or third-party repositories.
5. Describe the dataset, configuration, checkpoint, and metric threshold used to verify a behavioral change.

For substantial algorithm changes, open an issue first so that the proposed scope and evaluation protocol can be discussed.
