"""Regression checks for the Streamlit story page (no extra dependency).

Run from the project root or from frontend/:

    python frontend/check_app.py                   # offline, no API needed
    python frontend/check_app.py --live API_URL    # against the running stack

The offline check renders app.py headless with Streamlit's own AppTest while
the API is offline (nothing listens on API_URL) and fails when the page

* raises an exception (for example a Spotlight step naming a missing tab),
* shows a red error box instead of the "Live view paused" note,
* does not render all twelve story tabs,
* does not say under the title that the API is offline,
* builds the tool links from localhost although it runs in a codespace,
* shows a table row without any text (it looks like a rendering error),
* lets the tab row scroll on a narrow screen instead of wrapping (the last
  tabs would hide behind an arrow).

The live check (presentation readiness) needs the running stack with trained
champion models and fails when

* the Prediction tab offers no "Check" button under the test images,
* one click on "Check" does not show a verdict,
* that verdict stays visible after switching to another category,
* the Prometheus tab shows no "Average prediction time" or shows it above
  2 s (the bound of the AnomalyApiSlowPredictions alert).

Background: the first version showed raw connection errors when the API was
not running; the page must stay readable and calm without the API. The
presentation version opens every tool through its forwarded Codespaces
address and judges a test image with one click. The Prometheus tab once
showed an average of 30 s, because it averaged every route and a training
call takes minutes; it now averages /predict only.
"""

import os
import re
import sys
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent / "app.py"

# Port 9 (discard) is never served: every API call is refused at once.
OFFLINE_API_URL = "http://127.0.0.1:9"

# A made-up codespace: the tool links must be built from it.
CHECK_CODESPACE = "presentation-check"

# Same bound as the AnomalyApiSlowPredictions alert (monitoring/alert_rules.yml).
SLOW_PREDICTION_SECONDS = 2.0


def offline_problems():
    os.environ["API_URL"] = OFFLINE_API_URL
    os.environ["CODESPACE_NAME"] = CHECK_CODESPACE
    os.environ["GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN"] = "app.github.dev"

    app = AppTest.from_file(str(APP), default_timeout=60).run()

    problems = []

    if app.exception:
        problems.append(f"exception: {app.exception[0].value}")

    if app.error:
        problems.append(f"{len(app.error)} error box(es), first: {app.error[0].value}")

    if len(app.tabs) != 12:
        problems.append(f"expected 12 story tabs, found {len(app.tabs)}")

    if not any("API: offline" in info.value for info in app.info):
        problems.append("the page does not say that the API is offline")

    expected_link = f"https://{CHECK_CODESPACE}-8081.app.github.dev"
    link_urls = [button.proto.url for button in app.get("link_button")]

    if expected_link not in link_urls:
        problems.append(f"tool links are not built for the codespace (missing {expected_link})")

    # A table line made of "|" and spaces only: an empty header or row.
    empty_rows = [
        markdown.value
        for markdown in app.markdown
        if re.search(r"^\|(\s*\|)+\s*$", markdown.value, re.MULTILINE)
    ]

    if empty_rows:
        problems.append(f"{len(empty_rows)} table(s) with an empty row, first: {empty_rows[0][:60]!r}")

    page_styles = " ".join(node.proto.body for node in app.get("html"))

    if "flex-wrap: wrap" not in page_styles:
        problems.append("the tab row does not wrap: on a narrow screen the last tabs are hidden")

    return problems


def live_problems(api_url):
    os.environ["API_URL"] = api_url

    app = AppTest.from_file(str(APP), default_timeout=600).run()

    check_buttons = [
        button
        for button in app.button
        if str(button.key or "").startswith("predict_example_")
    ]

    if not check_buttons:
        return ["the Prediction tab shows no 'Check' button (API offline or no test images?)"]

    check_buttons[0].click().run()

    problems = []

    if app.exception:
        problems.append(f"exception after 'Check': {app.exception[0].value}")

    if app.error:
        problems.append(f"error box after 'Check': {app.error[0].value}")

    if not any("Verdict:" in markdown.value for markdown in app.markdown):
        problems.append("no verdict after one click on 'Check'")

    # A verdict belongs to the category it was made for: after a switch it
    # must not appear under the images of the new category.
    category_radio = next(radio for radio in app.radio if radio.label == "Category")
    other_category = next(option for option in ("bottle", "pill", "wood") if option != category_radio.value)
    category_radio.set_value(other_category).run()

    if any("Verdict:" in markdown.value for markdown in app.markdown):
        problems.append(f"the old verdict is still shown after switching to {other_category}")

    # The page keeps /metrics for 15 s: clear it so the average counts the
    # click above and is read from the API again.
    st.cache_data.clear()
    app.run()

    average = next(
        (metric.value for metric in app.metric if metric.label == "Average prediction time"),
        None,
    )

    if average is None:
        problems.append("the Prometheus tab shows no 'Average prediction time'")
    elif not average.endswith(" s"):
        problems.append(f"no /predict request counted after 'Check' (Average prediction time: {average})")
    elif float(average.removesuffix(" s")) > SLOW_PREDICTION_SECONDS:
        problems.append(
            f"Average prediction time {average} is above {SLOW_PREDICTION_SECONDS:.0f} s: "
            "run bash scripts/demo_setup.sh to warm the model, then check again"
        )

    return problems


def main():
    if len(sys.argv) == 1:
        mode, problems = "offline", offline_problems()
    elif len(sys.argv) == 3 and sys.argv[1] == "--live":
        mode, problems = "live", live_problems(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(2)

    if problems:
        print(f"FAILED ({mode})")
        for problem in problems:
            print(" -", problem)
        sys.exit(1)

    print(f"OK ({mode})")


if __name__ == "__main__":
    main()
