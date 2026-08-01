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
  tr.innerHTML = `
    <td>${fmtDate(rep.timestamp)}</td>
    <td>${rep.model}</td>
    <td>${rep.workload}</td>
    <td>${rep.prompts}</td>
    <td><span class="${gradeClass(rep.grade)}">${rep.composite.toFixed(1)} (${rep.grade})</span></td>
    <td class="${ttftColor}">${fmtPct(rep.ttft_improvement_pct)}</td>
    <td>${rep.hit_rate_pct.toFixed(0)}%</td>
    <td class="${tputColor}">${fmtPct(rep.throughput_gain_pct)}</td>
    <td class="${speedupColor}">${rep.speedup.toFixed(2)}x</td>
    <td>${rep.duration_baseline_sec.toFixed(0)}s / ${rep.duration_recycled_sec.toFixed(0)}s</td>
  `;
  return tr;
}

function renderSummary(reports) {
  document.getElementById("stat-count").textContent = reports.length;
  if (reports.length === 0) return;

  const best = reports.reduce((a, b) => a.composite > b.composite ? a : b);
  document.getElementById("stat-best").textContent =
    `${best.composite.toFixed(1)} (${best.model})`;

  const meanTtft = reports.reduce((s, r) => s + (r.ttft_improvement_pct || 0), 0) / reports.length;
  document.getElementById("stat-ttft").textContent = fmtPct(meanTtft);

  const meanHit = reports.reduce((s, r) => s + (r.hit_rate_pct || 0), 0) / reports.length;
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
    tbody.appendChild(renderRow(rep));
  }
}

main().catch((e) => {
  console.error(e);
});
