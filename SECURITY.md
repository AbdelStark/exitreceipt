# Security policy

ExitReceipt is an experimental classification tool. A `yes` prediction is not
proof that a task was completed and must not trigger sending, purchasing,
merging, deployment, or other consequential actions without an authoritative
tool receipt and the required human approval.

The public site loads a checked-in synthetic report; it has no input form,
server-side inference, or analytics. Local inference downloads model files
from the pinned upstream repository and processes text on your machine. Do not
paste secrets or private traces into public issues or pull requests.

## Reporting a vulnerability

Use GitHub's **Report a vulnerability** button under this repository's
Security → Advisories page. Include a minimal reproduction, affected commit,
and impact. Please keep exploit details private while the report is triaged.
For ordinary bugs without sensitive details, open a GitHub issue.

Only the current `main` branch receives fixes. No production safety guarantee
or response-time SLA is implied by this research repository.
