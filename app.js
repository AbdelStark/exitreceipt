const $ = (id) => document.getElementById(id);
let report;
let currentId;
let study = "v2";
const FILTERS = new Set(["all", "changed", "base-error", "tuned-error", "false-complete"]);
const STUDIES = { v2: "./results/v2-selected.json", pilot: "./results/pilot.json" };

function formatPct(value) { return `${(value * 100).toFixed(1)}%`; }
function verdict(value) { return value === "yes" ? "DONE" : "NOT DONE"; }

function renderDetail(row) {
  const panel = $("case-detail");
  panel.replaceChildren();
  if (!row) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No cases match this filter.";
    panel.append(empty);
    return;
  }
  const top = document.createElement("div"); top.className = "detail-top";
  const id = document.createElement("span"); id.textContent = row.id.toUpperCase();
  const domain = document.createElement("span"); domain.className = "detail-domain"; domain.textContent = row.domain;
  top.append(id, domain);
  const label = document.createElement("p"); label.className = "detail-title"; label.textContent = "THE REQUESTED OUTCOME";
  const goal = document.createElement("h3"); goal.className = "detail-goal"; goal.textContent = row.goal;
  const trace = document.createElement("div"); trace.className = "trace";
  const traceLabel = document.createElement("span"); traceLabel.className = "trace-label"; traceLabel.textContent = "LAST OBSERVED STATE";
  const traceText = document.createElement("span"); traceText.textContent = row.trace;
  trace.append(traceLabel, traceText);
  const verdicts = document.createElement("div"); verdicts.className = "verdicts";
  for (const [name, value] of [["GROUND TRUTH", row.truth], ["BASE", row.base], ["TUNED", row.tuned]]) {
    const card = document.createElement("div");
    card.className = `verdict ${name === "GROUND TRUTH" ? "truth" : value !== row.truth ? "wrong" : ""}`;
    const small = document.createElement("div"); small.className = "verdict-label"; small.textContent = name;
    const answer = document.createElement("div"); answer.className = "verdict-value"; answer.textContent = verdict(value);
    card.append(small, answer); verdicts.append(card);
  }
  const note = document.createElement("p"); note.className = "detail-note";
  note.textContent = study === "v2"
    ? "The label is WorkBench's sandbox verdict. The released action log omits tool results and final state, so some outcomes cannot be determined from this view alone. Spans are unverified model candidates."
    : "Ground truth follows the requested goal and stated tool outcome. Extracted spans are model candidates; they are not independently verified receipts.";
  const evidencePanel = document.createElement("section"); evidencePanel.className = "evidence-panel";
  const evidenceTitle = document.createElement("h4"); evidenceTitle.className = "evidence-title"; evidenceTitle.textContent = "CANDIDATE EVIDENCE SPANS";
  const columns = document.createElement("div"); columns.className = "evidence-columns";
  for (const [name, spans] of [["BASE", row.base_spans || []], ["TUNED", row.tuned_spans || []]]) {
    const column = document.createElement("div"); column.className = "evidence-column";
    const columnLabel = document.createElement("span"); columnLabel.className = "evidence-column-label"; columnLabel.textContent = name;
    column.append(columnLabel);
    if (!spans.length) {
      const empty = document.createElement("span"); empty.className = "evidence-item"; empty.textContent = "No span returned"; column.append(empty);
    }
    for (const span of spans) {
      const item = document.createElement("span"); item.className = "evidence-item";
      const kind = document.createElement("span"); kind.className = "evidence-kind"; kind.textContent = span.kind;
      item.append(kind, document.createTextNode(span.text)); column.append(item);
    }
    columns.append(column);
  }
  evidencePanel.append(evidenceTitle, columns);
  if (row.gold_evidence) {
    const gold = document.createElement("p"); gold.className = "gold-evidence";
    const goldLabel = document.createElement("span"); goldLabel.className = "gold-label"; goldLabel.textContent = "ANNOTATED DECISIVE PHRASE";
    gold.append(goldLabel, document.createTextNode(row.gold_evidence.evidence)); evidencePanel.append(gold);
  }
  panel.append(top, label, goal, trace, verdicts, evidencePanel, note);
  if (study === "v2") {
    const counterpartId = row.id.endsWith("-yes") ? row.id.slice(0, -4) + "-no" : row.id.slice(0, -3) + "-yes";
    if (report.rows.some((candidate) => candidate.id === counterpartId)) {
      const counterpart = document.createElement("button");
      counterpart.className = "pair-link";
      counterpart.type = "button";
      counterpart.textContent = "VIEW THE OTHER RUN OF THIS TASK ↗";
      counterpart.addEventListener("click", () => {
        $("filter").value = "all";
        currentId = counterpartId;
        renderList();
      });
      panel.append(counterpart);
    }
  }
}

function filteredRows() {
  const filter = $("filter").value;
  return report.rows.filter((row) => {
    if (filter === "changed") return row.base !== row.tuned;
    if (filter === "base-error") return row.base !== row.truth;
    if (filter === "tuned-error") return row.tuned !== row.truth;
    if (filter === "false-complete") return row.truth === "no" && (row.base === "yes" || row.tuned === "yes");
    return true;
  });
}

function renderList() {
  const rows = filteredRows();
  $("case-count").textContent = `${rows.length} / ${report.rows.length} CASES`;
  if (!rows.some((row) => row.id === currentId)) currentId = rows[0]?.id;
  const list = $("case-list"); list.replaceChildren();
  for (const row of rows) {
    const button = document.createElement("button"); button.className = `case-item ${row.id === currentId ? "active" : ""}`;
    button.type = "button"; button.setAttribute("role", "option"); button.setAttribute("aria-selected", String(row.id === currentId));
    const top = document.createElement("div"); top.className = "case-item-top";
    const id = document.createElement("span"); id.className = "case-id"; id.textContent = row.id.toUpperCase();
    const tag = document.createElement("span"); tag.className = "case-tag";
    tag.textContent = row.base !== row.tuned ? "CHANGED" : row.base !== row.truth ? "BOTH WRONG" : "AGREED";
    const title = document.createElement("div"); title.className = "case-item-title"; title.textContent = row.goal;
    top.append(id, tag); button.append(top, title);
    button.addEventListener("click", () => { currentId = row.id; renderList(); });
    list.append(button);
  }
  renderDetail(rows.find((row) => row.id === currentId));
  const url = new URL(window.location.href);
  if (study !== "v2") url.searchParams.set("study", study);
  else url.searchParams.delete("study");
  if (currentId) url.searchParams.set("case", currentId);
  else url.searchParams.delete("case");
  if ($("filter").value !== "all") url.searchParams.set("filter", $("filter").value);
  else url.searchParams.delete("filter");
  window.history.replaceState(null, "", url);
}

async function main() {
  try {
    const params = new URLSearchParams(window.location.search);
    const requestedCase = params.get("case");
    study = params.get("study") === "pilot" || (!params.has("study") && /^te\d+$/.test(requestedCase || "")) ? "pilot" : "v2";
    $("study").value = study;
    await loadStudy(requestedCase, params.get("filter"));
    $("study").addEventListener("change", async () => {
      study = $("study").value;
      $("filter").value = "all";
      await loadStudy(null, null);
    });
    $("filter").addEventListener("change", renderList);
  } catch (error) {
    showError(error);
  }
}

function showError(error) {
  $("error").hidden = false;
  $("error").textContent = error instanceof Error ? error.message : String(error);
  $("case-detail").textContent = "Open the repository for the recorded evaluation report.";
}

async function loadStudy(caseId, filter) {
  try {
    const response = await fetch(STUDIES[study], { cache: "no-store" });
    if (!response.ok) throw new Error(`Report unavailable (HTTP ${response.status}). Run the evaluation command first.`);
    report = await response.json();
    if (report.schema_version !== 1 || report.split !== "test" || !Array.isArray(report.rows) || !report.rows.length || !report.base || !report.tuned) throw new Error("Report schema is incomplete or is not the final test run.");
    $("error").hidden = true;
    if (FILTERS.has(filter)) $("filter").value = filter;
    currentId = caseId;
    $("sample-count").textContent = `${report.rows.length} HELD-OUT CASES / ${study === "v2" ? "WORKBENCH" : "AUTHORED PILOT"}`;
    $("base-accuracy").textContent = formatPct(report.base.accuracy);
    $("tuned-accuracy").textContent = formatPct(report.tuned.accuracy);
    $("base-false").textContent = `${report.base.false_complete} FALSE COMPLETION ${report.base.false_complete === 1 ? "CALL" : "CALLS"}`;
    $("tuned-false").textContent = `${report.tuned.false_complete} FALSE COMPLETION ${report.tuned.false_complete === 1 ? "CALL" : "CALLS"}`;
    const delta = (report.tuned.accuracy - report.base.accuracy) * 100;
    $("delta").textContent = `${delta > 0 ? "+" : ""}${delta.toFixed(1)}`;
    $("changed-count").textContent = `${report.rows.filter((row) => row.base !== row.tuned).length} CHANGED DECISIONS`;
    $("evidence-score").hidden = true;
    if (report.evidence?.base?.annotated_n && report.evidence?.tuned?.annotated_n) {
      const b = report.evidence.base; const t = report.evidence.tuned;
      $("evidence-score").textContent = `ANNOTATED EVIDENCE SPANS · ≥50% GOLD OVERLAP: ${b.half_gold_span_and_type_hits}/${b.annotated_n} BASE → ${t.half_gold_span_and_type_hits}/${t.annotated_n} TUNED`;
      $("evidence-score").hidden = false;
    }
    $("caveat").textContent = study === "v2"
      ? "Sandbox-scored WorkBench action logs, held out by task template. Tool results and final state are absent from the released action view. This measures outcome prediction from partial evidence, not verification of a live agent."
      : "Authored synthetic English cases. The tuned model corrected two decisions and regressed on two others. A verdict is a review signal; the underlying receipt remains the source of truth.";
    $("scores").hidden = false;
    renderList();
  } catch (error) {
    showError(error);
  }
}
main();
