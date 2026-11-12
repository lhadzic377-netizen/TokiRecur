// Toki Recur dashboard frontend

async function loadReports() {
  const r = await fetch("/api/reports");
  return r.json();
}

function fmtDate(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toISOString().slice(0, 16).replace("T", " ");
}

function gradeClass(grade) {
  if (!grade || grade === "N/A") return "";
  return `grade-${grade}`;
}

function fmtPct(v) {
  if (v == null) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(1)}%`;
}

function colorForDelta(v) {
  if (v == null) return "";
  if (v > 20) return "good";
  if (v < -10) return "bad";
  return "neutral";
}

function renderRow(rep) {
  const tr = document.createElement("tr");
  const ttftColor = colorForDelta(rep.ttft_improvement_pct);
  const tputColor = colorForDelta(rep.throughput_gain_pct);
  const speedupColor = rep.speedup > 1.05 ? "good" : rep.speedup < 0.95 ? "bad" : "neutral";

  // Every value here comes from a report JSON file: the model is a
  // GGUF file name, the workload is an arbitrary string from a shared
  // workload file, and a report can be dropped into the reports
  // directory from anywhere. Building the row with innerHTML let any
  // of them run script in the dashboard's origin, so the cells are
  // text and the markup is ours.
  const cell = (text, className) => {
    const td = document.createElement("td");
    td.textContent = text;
    if (className) td.className = className;
    return td;
  };

  tr.appendChild(cell(fmtDate(rep.timestamp)));
  tr.appendChild(cell(String(rep.model)));
  tr.appendChild(cell(String(rep.workload)));
  tr.appendChild(cell(String(rep.prompts)));

  const gradeCell = document.createElement("td");
  const gradeSpan = document.createElement("span");
  const gradeCls = gradeClass(rep.grade);
  if (gradeCls) gradeSpan.className = gradeCls;
  gradeSpan.textContent = `${rep.composite.toFixed(1)} (${rep.grade})`;
  gradeCell.appendChild(gradeSpan);
  tr.appendChild(gradeCell);

  tr.appendChild(cell(fmtPct(rep.ttft_improvement_pct), ttftColor));
  tr.appendChild(cell(`${rep.hit_rate_pct.toFixed(0)}%`));
  tr.appendChild(cell(fmtPct(rep.throughput_gain_pct), tputColor));
  tr.appendChild(cell(`${rep.speedup.toFixed(2)}x`, speedupColor));
  tr.appendChild(cell(`${rep.duration_baseline_sec.toFixed(0)}s / ${rep.duration_recycled_sec.toFixed(0)}s`));
  return tr;
}

function renderSummary(reports) {
  document.getElementById("stat-count").textContent = reports.length;
  if (reports.length === 0) return;

  // One malformed report must not blank the whole dashboard: the
  // arithmetic below used to throw on the first null and the table
  // never rendered at all.
  const num = (v) => (typeof v === "number" && isFinite(v) ? v : 0);
  const composite = (r) => num(r.composite);

  const best = reports.reduce((a, b) => composite(a) > composite(b) ? a : b);
  document.getElementById("stat-best").textContent =
    `${composite(best).toFixed(1)} (${best.model})`;

  const meanTtft = reports.reduce((s, r) => s + num(r.ttft_improvement_pct), 0) / reports.length;
  document.getElementById("stat-ttft").textContent = fmtPct(meanTtft);

  const meanHit = reports.reduce((s, r) => s + num(r.hit_rate_pct), 0) / reports.length;
  document.getElementById("stat-hit").textContent = `${meanHit.toFixed(0)}%`;
}

async function main() {
  const reports = await loadReports();
  renderSummary(reports);

  const tbody = document.getElementById("reports-body");
  const empty = document.getElementById("empty-msg");
  if (reports.length === 0) {
    empty.style.display = "block";
    return;
  }
  empty.style.display = "none";
  for (const rep of reports) {
    // One report that cannot be rendered must not take the table with
    // it: the rest are still worth showing.
    try {
      tbody.appendChild(renderRow(rep));
    } catch (e) {
      console.error("skipping report", rep && rep.file, e);
    }
  }
}

main().catch((e) => {
  console.error(e);
});
