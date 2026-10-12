# Hackathon notes

## neatlogs usage

`agent.py` initializes a `leakhound-audit` workflow and traces the workflow,
Gemini calls, and tool calls. Tool-call spans use the `TOOL` kind.

TODO: Add neatlogs dashboard link.
TODO: Add screenshot of an audit trace.

## Entire

TODO: Add the verified description of Entire's role and a link.
TODO: Add screenshot, if applicable.

## cfo.ai

TODO: Add the verified description of cfo.ai's role and a link.
TODO: Add screenshot, if applicable.

## failure-recovery run

The model-call helper retries rate-limit or service-unavailable errors and
then attempts the configured fallback model. This describes the code path; it
does not claim that a live recovery run was performed.

TODO: Add date and sanitized evidence for a verified failure-recovery run.
TODO: Add a link or screenshot.

## results

The checked-in reports record a planted Telco column leak with accuracy falling
from 0.9972 to 0.78 after removing `cancellation_reason`. The clean dataset's
current report records `no_column_leak_found` with baseline accuracy 0.78.
Saved benchmark summaries report the leaky verdict in all three runs; clean
data was reported as no column leak in three runs in one summary, and in two
runs with one error in another.

TODO: Add the results/demo link.
TODO: Add screenshot of the current report.

## limitations

The measured checks currently address individual-column signals and exact
train/test duplicate feature rows using `train_demo.py`. The Telco leak is
planted, the sample is small, and Gemini's free-tier quota is limited. The AST
preprocessing-order check is a static suspicion, not a measured finding.
Arbitrary notebooks and custom training pipelines are unsupported.
