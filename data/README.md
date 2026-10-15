# Data

The training data is the UCI "Diabetes 130-US Hospitals for Years
1999-2008" dataset (101,766 encounters). It is not committed to this
repo; the raw CSV is gitignored because UCI distributes it under its own
terms.

Download it from the UCI Machine Learning Repository:

    https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Direct CSV (the same file the UCI API returns):

    curl -o data/diabetic_data.csv \
      "https://archive.ics.uci.edu/static/public/296/data.csv"

Then train with `make train`.

The committed sample at `tests/fixtures/sample_encounters.csv` is 360
rows drawn at random from the real download, stratified by the
readmission label, so the test suite can train a model fully offline.
