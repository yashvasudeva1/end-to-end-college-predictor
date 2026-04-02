/* ═══════════════════════════════════════════
   JoSAA College Predictor - Frontend Logic
   ═══════════════════════════════════════════ */

// ──────────────── INTERACTIVE BACKGROUND ────────────────
// Organic floating orbs that drift and pulse - no mouse-connected graph
(function initBackground() {
  const canvas = document.getElementById("bg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let W, H;
  const orbs = [];
  const ORB_COUNT = 18;

  function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  window.addEventListener("resize", resize);
  resize();

  // Track mouse for subtle glow interaction
  let mx = W / 2, my = H / 2;
  document.addEventListener("mousemove", (e) => {
    mx = e.clientX;
    my = e.clientY;
  });

  class Orb {
    constructor() {
      this.reset();
    }
    reset() {
      this.x = Math.random() * W;
      this.y = Math.random() * H;
      this.r = 80 + Math.random() * 200;
      this.vx = (Math.random() - 0.5) * 0.35;
      this.vy = (Math.random() - 0.5) * 0.35;
      this.phase = Math.random() * Math.PI * 2;
      this.speed = 0.003 + Math.random() * 0.005;
      // Warm hues: orange / amber / teal only
      const palette = [
        { r: 232, g: 148, b: 58 },   // accent orange
        { r: 196, g: 117, b: 32 },   // darker amber
        { r: 56, g: 178, b: 172 },   // teal
        { r: 245, g: 183, b: 108 },  // light amber
        { r: 92, g: 212, b: 206 },   // light teal
        { r: 200, g: 130, b: 60 },   // warm brown
      ];
      this.color = palette[Math.floor(Math.random() * palette.length)];
      this.baseAlpha = 0.015 + Math.random() * 0.03;
    }
    update(t) {
      this.x += this.vx;
      this.y += this.vy;
      this.phase += this.speed;

      // Wrap around
      if (this.x < -this.r) this.x = W + this.r;
      if (this.x > W + this.r) this.x = -this.r;
      if (this.y < -this.r) this.y = H + this.r;
      if (this.y > H + this.r) this.y = -this.r;

      // Pulse radius
      this.currentR = this.r + Math.sin(this.phase) * 20;

      // Boost alpha near mouse
      const dx = this.x - mx;
      const dy = this.y - my;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const proximity = Math.max(0, 1 - dist / 500);
      this.alpha = this.baseAlpha + proximity * 0.04;
    }
    draw() {
      const grad = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.currentR);
      const { r, g, b } = this.color;
      grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${this.alpha})`);
      grad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.currentR, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  for (let i = 0; i < ORB_COUNT; i++) orbs.push(new Orb());

  let t = 0;
  function animate() {
    t++;
    ctx.clearRect(0, 0, W, H);
    for (const orb of orbs) {
      orb.update(t);
      orb.draw();
    }
    requestAnimationFrame(animate);
  }
  animate();
})();


// ──────────────── NAVIGATION ────────────────

const pages = document.querySelectorAll(".page");
const navItems = document.querySelectorAll(".nav-item");
const sidebar = document.getElementById("sidebar");
const overlay = document.getElementById("sidebar-overlay");
const burger = document.getElementById("burger-btn");

function showPage(pageId) {
  pages.forEach((p) => {
    p.classList.remove("active", "fade-up");
    p.style.display = "none";
  });
  navItems.forEach((n) => n.classList.remove("active"));

  const target = document.getElementById(`page-${pageId}`);
  if (target) {
    target.style.display = "block";
    // Force reflow before adding animation class
    void target.offsetWidth;
    target.classList.add("active", "fade-up");
  }

  const activeNav = document.querySelector(`.nav-item[data-page="${pageId}"]`);
  if (activeNav) activeNav.classList.add("active");

  // Close mobile sidebar
  sidebar.classList.remove("open");
  overlay.classList.remove("visible");

  // Load data for this page
  loadPageData(pageId);
}

navItems.forEach((item) => {
  item.addEventListener("click", () => showPage(item.dataset.page));
});

burger.addEventListener("click", () => {
  sidebar.classList.toggle("open");
  overlay.classList.toggle("visible");
});
overlay.addEventListener("click", () => {
  sidebar.classList.remove("open");
  overlay.classList.remove("visible");
});


// ──────────────── DATA LOADING ────────────────

const cache = {};
const chartInstances = {};

async function fetchJSON(url) {
  if (cache[url]) return cache[url];
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  cache[url] = data;
  return data;
}

// Destroy old chart before creating new one
function getCtx(canvasId) {
  if (chartInstances[canvasId]) {
    chartInstances[canvasId].destroy();
    delete chartInstances[canvasId];
  }
  return document.getElementById(canvasId).getContext("2d");
}

function storeChart(canvasId, chart) {
  chartInstances[canvasId] = chart;
}


// Chart.js default styling
Chart.defaults.color = "#9b978f";
Chart.defaults.borderColor = "rgba(255,255,255,0.04)";
Chart.defaults.font.family = "'DM Sans', system-ui, sans-serif";
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;

const ACCENT = "#e8943a";
const ACCENT_LIGHT = "#f5b76c";
const TEAL = "#38b2ac";
const TEAL_LIGHT = "#5cd4ce";

// Year color palette (warm, no purple)
const YEAR_COLORS = [
  "#e8943a",  // orange
  "#38b2ac",  // teal
  "#d69e2e",  // gold
  "#e05a3a",  // coral red
  "#5cd4ce",  // light teal
  "#c47520",  // dark amber
];


// ──────────────── PAGE LOADERS ────────────────

const loadedPages = new Set();

async function loadPageData(pageId) {
  if (loadedPages.has(pageId)) return;

  try {
    switch (pageId) {
      case "home":
        await loadHome();
        break;
      case "data-overview":
        await loadDataOverview();
        break;
      case "data-analysis":
        await loadDataAnalysis();
        break;
      case "trends":
        await loadTrends();
        break;
      case "performance":
        await loadPerformance();
        break;
      // predict and methodology don't need preloading
    }
    loadedPages.add(pageId);
  } catch (err) {
    console.error(`Error loading ${pageId}:`, err);
  }
}


// ── HOME ──
async function loadHome() {
  const stats = await fetchJSON("/api/stats");
  animateNumber("stat-institutes", stats.institutes);
  animateNumber("stat-branches", stats.branches);
  animateNumber("stat-years", stats.years);
  animateNumber("stat-records", stats.records, true);
}

function animateNumber(id, target, formatted = false) {
  const el = document.getElementById(id);
  const duration = 1200;
  const start = performance.now();

  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // ease-out
    const val = Math.round(eased * target);
    el.textContent = formatted ? val.toLocaleString() : val;
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}


// ── DATA OVERVIEW ──
async function loadDataOverview() {
  // Sample table
  const stats = await fetchJSON("/api/stats");
  const overview = await fetchJSON("/api/overview-trend");

  // Build sample table (use overview-trend data as a quick table)
  const wrapper = document.getElementById("sample-table-wrapper");
  // We'll fetch competitive institutes as a proxy for sample data
  const instData = await fetchJSON("/api/analysis/competitive-institutes");
  const branchData = await fetchJSON("/api/analysis/competitive-branches");

  let tableHTML = `<table class="data-table">
    <thead><tr><th>Rank</th><th>Institute</th><th>Median Closing Rank</th></tr></thead>
    <tbody>`;
  instData.forEach((row, i) => {
    tableHTML += `<tr><td>${i + 1}</td><td>${row.institute}</td><td>${Math.round(row.close_rank).toLocaleString()}</td></tr>`;
  });
  tableHTML += `</tbody></table>`;
  wrapper.innerHTML = tableHTML;

  // Overview trend chart
  const years = overview.map((d) => d.year);
  const medians = overview.map((d) => d.median);
  const q25 = overview.map((d) => d.q25);
  const q75 = overview.map((d) => d.q75);

  const ctx = getCtx("chart-overview-trend");
  const chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: years,
      datasets: [
        {
          label: "Median",
          data: medians,
          borderColor: ACCENT,
          backgroundColor: "rgba(232,148,58,0.1)",
          borderWidth: 2.5,
          pointRadius: 5,
          pointBackgroundColor: ACCENT,
          tension: 0.3,
          fill: false,
        },
        {
          label: "Q25 (top quartile)",
          data: q25,
          borderColor: TEAL,
          borderWidth: 1.5,
          borderDash: [6, 4],
          pointRadius: 3,
          pointBackgroundColor: TEAL,
          tension: 0.3,
          fill: false,
        },
        {
          label: "Q75 (bottom quartile)",
          data: q75,
          borderColor: TEAL_LIGHT,
          borderWidth: 1.5,
          borderDash: [6, 4],
          pointRadius: 3,
          pointBackgroundColor: TEAL_LIGHT,
          tension: 0.3,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          reverse: true,
          title: { display: true, text: "Closing Rank", color: "#9b978f" },
          grid: { color: "rgba(255,255,255,0.03)" },
        },
        x: {
          title: { display: true, text: "Year", color: "#9b978f" },
          grid: { display: false },
        },
      },
      interaction: { mode: "index", intersect: false },
      plugins: {
        tooltip: {
          backgroundColor: "rgba(14,17,24,0.95)",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
          titleColor: "#e2e0dc",
          bodyColor: "#9b978f",
        },
      },
    },
  });
  storeChart("chart-overview-trend", chart);
}


// ── DATA ANALYSIS ──
async function loadDataAnalysis() {
  const [instStab, branchVol, yearTrend, roundTrend, compInst, compBranch] =
    await Promise.all([
      fetchJSON("/api/analysis/institute-stability"),
      fetchJSON("/api/analysis/branch-volatility"),
      fetchJSON("/api/analysis/year-trend"),
      fetchJSON("/api/analysis/round-trend"),
      fetchJSON("/api/analysis/competitive-institutes"),
      fetchJSON("/api/analysis/competitive-branches"),
    ]);

  // Institute stability
  createHorizontalBar("chart-inst-stability", instStab, "institute", "std_dev", TEAL, "Std. Dev of Closing Rank");

  // Branch volatility
  createHorizontalBar("chart-branch-volatility", branchVol, "branch", "std_dev", "#e05a3a", "Std. Dev of Closing Rank");

  // Year trend
  createLineChart("chart-year-trend", yearTrend.map((d) => d.year), yearTrend.map((d) => d.close_rank), "Median Closing Rank", ACCENT, true);

  // Round trend
  createLineChart("chart-round-trend", roundTrend.map((d) => `Round ${d.round}`), roundTrend.map((d) => d.close_rank), "Median Closing Rank", TEAL, true);

  // Competitive institutes
  createHorizontalBar("chart-comp-inst", compInst, "institute", "close_rank", ACCENT, "Median Closing Rank", true);

  // Competitive branches
  createHorizontalBar("chart-comp-branch", compBranch, "branch", "close_rank", TEAL, "Median Closing Rank", true);
}


function createHorizontalBar(canvasId, data, labelKey, valueKey, color, xLabel, reverseX = false) {
  const ctx = getCtx(canvasId);

  // Truncate long labels
  const labels = data.map((d) => {
    const l = d[labelKey];
    return l.length > 45 ? l.substring(0, 42) + "…" : l;
  });

  const chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          data: data.map((d) => Math.round(d[valueKey])),
          backgroundColor: color + "55",
          borderColor: color,
          borderWidth: 1.5,
          borderRadius: 4,
          barThickness: 22,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          reverse: reverseX,
          title: { display: true, text: xLabel, color: "#9b978f" },
          grid: { color: "rgba(255,255,255,0.03)" },
        },
        y: {
          grid: { display: false },
          ticks: {
            font: { size: 11 },
            callback: function (val) {
              const label = this.getLabelForValue(val);
              return label.length > 30 ? label.substring(0, 28) + "…" : label;
            },
          },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(14,17,24,0.95)",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
          callbacks: {
            title: (items) => data[items[0].dataIndex][labelKey],
          },
        },
      },
    },
  });
  storeChart(canvasId, chart);
}


function createLineChart(canvasId, labels, values, label, color, reverseY = false) {
  const ctx = getCtx(canvasId);
  const chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label,
          data: values,
          borderColor: color,
          backgroundColor: color + "18",
          borderWidth: 2.5,
          pointRadius: 5,
          pointBackgroundColor: color,
          tension: 0.3,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          reverse: reverseY,
          title: { display: true, text: label, color: "#9b978f" },
          grid: { color: "rgba(255,255,255,0.03)" },
        },
        x: { grid: { display: false } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(14,17,24,0.95)",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
        },
      },
    },
  });
  storeChart(canvasId, chart);
}


// ── TABS ──
document.querySelectorAll(".tab-bar").forEach((bar) => {
  bar.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab-btn");
    if (!btn) return;

    bar.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");

    const tabId = btn.dataset.tab;
    const section = bar.closest("section");
    section.querySelectorAll(".tab-panel").forEach((p) => {
      p.classList.remove("active");
    });
    const panel = document.getElementById(tabId);
    if (panel) panel.classList.add("active");
  });
});


// ── CLOSING RANK TRENDS ──
let closingTrendChart = null;

async function loadTrends() {
  const institutes = await fetchJSON("/api/institutes");
  const select = document.getElementById("trend-institute");
  select.innerHTML = institutes
    .map((inst) => `<option value="${inst}">${inst}</option>`)
    .join("");

  select.addEventListener("change", loadBranches);
  document.getElementById("trend-branch").addEventListener("change", loadClosingTrend);

  // Load first institute's branches
  if (institutes.length > 0) {
    await loadBranches();
  }
}

async function loadBranches() {
  const inst = document.getElementById("trend-institute").value;
  const branches = await fetchJSON(`/api/branches?institute=${encodeURIComponent(inst)}`);
  // Clear cache for branches since it changes per institute
  delete cache[`/api/branches?institute=${encodeURIComponent(inst)}`];

  const branchSelect = document.getElementById("trend-branch");
  branchSelect.innerHTML = branches
    .map((b) => `<option value="${b}">${b}</option>`)
    .join("");

  if (branches.length > 0) {
    await loadClosingTrend();
  }
}

async function loadClosingTrend() {
  const inst = document.getElementById("trend-institute").value;
  const branch = document.getElementById("trend-branch").value;

  if (!inst || !branch) return;

  const url = `/api/closing-trend?institute=${encodeURIComponent(inst)}&branch=${encodeURIComponent(branch)}`;
  // Don't cache trend data since selections change
  const res = await fetch(url);
  const data = await res.json();

  const title = document.getElementById("trend-chart-title");
  title.textContent = `Round-wise Closing Rank - ${inst.length > 50 ? inst.substring(0, 47) + '…' : inst}`;

  if (data.length === 0) {
    if (closingTrendChart) {
      closingTrendChart.destroy();
      closingTrendChart = null;
    }
    return;
  }

  // Group by year
  const yearMap = {};
  data.forEach((d) => {
    if (!yearMap[d.year]) yearMap[d.year] = [];
    yearMap[d.year].push(d);
  });

  const years = Object.keys(yearMap).sort();
  const allRounds = [...new Set(data.map((d) => d.round))].sort((a, b) => a - b);

  const datasets = years.map((year, i) => {
    const yearData = yearMap[year];
    const roundMap = {};
    yearData.forEach((d) => (roundMap[d.round] = d.close_rank));

    return {
      label: year.toString(),
      data: allRounds.map((r) => roundMap[r] || null),
      borderColor: YEAR_COLORS[i % YEAR_COLORS.length],
      backgroundColor: YEAR_COLORS[i % YEAR_COLORS.length] + "18",
      borderWidth: 2,
      pointRadius: 4,
      pointBackgroundColor: YEAR_COLORS[i % YEAR_COLORS.length],
      tension: 0.3,
      spanGaps: true,
    };
  });

  const ctx = getCtx("chart-closing-trend");
  closingTrendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: allRounds.map((r) => `Round ${r}`),
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          reverse: true,
          title: { display: true, text: "Closing Rank", color: "#9b978f" },
          grid: { color: "rgba(255,255,255,0.03)" },
        },
        x: {
          title: { display: true, text: "JoSAA Round", color: "#9b978f" },
          grid: { display: false },
        },
      },
      interaction: { mode: "index", intersect: false },
      plugins: {
        tooltip: {
          backgroundColor: "rgba(14,17,24,0.95)",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
        },
      },
    },
  });
  storeChart("chart-closing-trend", closingTrendChart);
}


// ── PREDICT ──
document.getElementById("btn-predict").addEventListener("click", runPrediction);
document.getElementById("predict-rank").addEventListener("keydown", (e) => {
  if (e.key === "Enter") runPrediction();
});

async function runPrediction() {
  const rankInput = document.getElementById("predict-rank");
  const rank = parseInt(rankInput.value);
  const resultsDiv = document.getElementById("predict-results");
  const btn = document.getElementById("btn-predict");

  if (!rank || rank < 1) {
    resultsDiv.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠</div><p>Please enter a valid rank (must be 1 or higher).</p></div>`;
    return;
  }

  btn.disabled = true;
  btn.innerHTML = "<span class='spinner' style='width:18px;height:18px;border-width:2px;'></span> Predicting...";
  resultsDiv.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Running predictions for rank ${rank.toLocaleString()}...</div>`;

  try {
    const res = await fetch(`/api/predict?rank=${rank}`);
    const data = await res.json();

    if (data.error) {
      resultsDiv.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠</div><p>${data.error}</p></div>`;
      return;
    }

    renderPredictions(data, rank);
  } catch (err) {
    resultsDiv.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠</div><p>Error running prediction. Please try again.</p></div>`;
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = "<span>⎈</span> Predict";
  }
}

function renderPredictions(response, rank) {
  const resultsDiv = document.getElementById("predict-results");
  const data = response.results;
  const closeMae = response.close_mae;
  const openMae = response.open_mae;
  const groups = { Safe: [], Moderate: [], Risky: [], "Very Risky": [] };

  data.forEach((row) => {
    if (groups[row.chance]) groups[row.chance].push(row);
  });

  const chanceConfig = {
    Safe: { cls: "safe", icon: "\u2713", desc: "High likelihood of admission" },
    Moderate: { cls: "moderate", icon: "\u25D0", desc: "Reasonable chances, consider as options" },
    Risky: { cls: "risky", icon: "\u25B3", desc: "Lower probability, keep as backup" },
    "Very Risky": { cls: "very-risky", icon: "\u2715", desc: "Very low probability of admission" },
  };

  let html = `<div class="results-section">`;
  html += `<div style="margin-bottom:20px;color:var(--text-dim);font-size:0.88rem;">Showing predictions for rank <strong style="color:var(--accent)">${rank.toLocaleString()}</strong> \u2014 ${data.length.toLocaleString()} college-branch-round combinations analysed.</div>`;

  html += `<div class="error-info-banner">
    <span>\u24D8</span>
    <span>Predicted ranks have an estimated error margin of <strong>\u00b1${closeMae.toLocaleString()}</strong> for closing rank and <strong>\u00b1${openMae.toLocaleString()}</strong> for opening rank, based on test-set performance (year 2024). The range shown below reflects this uncertainty.</span>
  </div>`;

  for (const [chance, config] of Object.entries(chanceConfig)) {
    const items = groups[chance];
    html += `<div class="chance-group">`;
    html += `<div class="chance-label ${config.cls}">${config.icon} ${chance} <span class="result-count">(${items.length})</span></div>`;

    if (items.length === 0) {
      html += `<div class="no-data-msg">No colleges fall under ${chance} category for your rank.</div>`;
    } else {
      html += `<div class="table-scroll"><table class="result-table">
        <thead><tr><th>Round</th><th>Institute</th><th>Branch</th><th>Predicted Opening</th><th>Predicted Closing</th></tr></thead>
        <tbody>`;
      items.slice(0, 150).forEach((row) => {
        html += `<tr>
          <td><span class="round-badge">${row.round}</span></td>
          <td>${row.institute}</td>
          <td>${row.branch}</td>
          <td>
            ${row.pred_open.toLocaleString()}
            <span class="error-range">${row.open_low.toLocaleString()} \u2013 ${row.open_high.toLocaleString()}</span>
          </td>
          <td>
            ${row.pred_close.toLocaleString()}
            <span class="error-range">${row.close_low.toLocaleString()} \u2013 ${row.close_high.toLocaleString()}</span>
          </td>
        </tr>`;
      });
      if (items.length > 150) {
        html += `<tr><td colspan="5" style="text-align:center;color:var(--text-muted);font-size:0.82rem;">... and ${items.length - 150} more</td></tr>`;
      }
      html += `</tbody></table></div>`;
    }
    html += `</div>`;
  }

  html += `</div>`;
  resultsDiv.innerHTML = html;
}


// \u2500\u2500 MODEL PERFORMANCE \u2500\u2500
async function loadPerformance() {
  const perf = await fetchJSON("/api/model-performance");

  // Helper: fill a comparison table row
  function fillRow(prefix, metricName, trainVal, testVal, isR2 = false) {
    const trainEl = document.getElementById(`${prefix}-train-${metricName}`);
    const testEl = document.getElementById(`${prefix}-test-${metricName}`);
    const diffEl = document.getElementById(`${prefix}-diff-${metricName}`);
    const assessEl = document.getElementById(`${prefix}-assess-${metricName}`);

    if (isR2) {
      trainEl.textContent = trainVal.toFixed(4);
      testEl.textContent = testVal.toFixed(4);
      const drop = trainVal - testVal;
      const sign = drop >= 0 ? "\u2193" : "\u2191";
      diffEl.textContent = `${sign} ${Math.abs(drop).toFixed(4)}`;

      if (drop < 0.05 && testVal > 0.85) {
        assessEl.innerHTML = `<span class="assess-pill good">Good Fit</span>`;
      } else if (drop < 0.15) {
        assessEl.innerHTML = `<span class="assess-pill ok">Acceptable</span>`;
      } else {
        assessEl.innerHTML = `<span class="assess-pill poor">Overfitting</span>`;
      }
    } else {
      trainEl.textContent = Math.round(trainVal).toLocaleString();
      testEl.textContent = Math.round(testVal).toLocaleString();
      const ratio = testVal / Math.max(trainVal, 1);
      const pctDiff = ((ratio - 1) * 100).toFixed(1);
      diffEl.textContent = `${ratio >= 1 ? '+' : ''}${pctDiff}%`;

      if (ratio < 1.3) {
        assessEl.innerHTML = `<span class="assess-pill good">Good</span>`;
      } else if (ratio < 2.0) {
        assessEl.innerHTML = `<span class="assess-pill ok">Acceptable</span>`;
      } else {
        assessEl.innerHTML = `<span class="assess-pill poor">High Gap</span>`;
      }
    }
  }

  // Closing Rank Model
  fillRow("close", "mae", perf.train.closing.mae, perf.test.closing.mae);
  fillRow("close", "rmse", perf.train.closing.rmse, perf.test.closing.rmse);
  fillRow("close", "r2", perf.train.closing.r2, perf.test.closing.r2, true);
  document.getElementById("close-train-size").textContent =
    `Train: ${perf.train.size.toLocaleString()} records (${perf.train.years}) \u00a0|\u00a0 Test: ${perf.test.size.toLocaleString()} records (${perf.test.years})`;

  // Opening Rank Model
  fillRow("open", "mae", perf.train.opening.mae, perf.test.opening.mae);
  fillRow("open", "rmse", perf.train.opening.rmse, perf.test.opening.rmse);
  fillRow("open", "r2", perf.train.opening.r2, perf.test.opening.r2, true);
  document.getElementById("open-train-size").textContent =
    `Train: ${perf.train.size.toLocaleString()} records (${perf.train.years}) \u00a0|\u00a0 Test: ${perf.test.size.toLocaleString()} records (${perf.test.years})`;

  // Model Fit Verdict
  const verdictDiv = document.getElementById("model-fit-verdict");
  const verdictIcon = document.getElementById("verdict-icon");
  const verdictTitle = document.getElementById("verdict-title");
  const verdictText = document.getElementById("verdict-text");
  verdictDiv.style.display = "block";

  const closeR2Drop = perf.train.closing.r2 - perf.test.closing.r2;
  const closeTestR2 = perf.test.closing.r2;
  const maeRatio = perf.test.closing.mae / Math.max(perf.train.closing.mae, 1);

  if (closeR2Drop < 0.05 && closeTestR2 > 0.85 && maeRatio < 1.5) {
    verdictIcon.textContent = "\u2705";
    verdictTitle.textContent = "Good Fit \u2014 The model generalises well";
    verdictTitle.style.color = "#68d391";
    verdictText.textContent = `Train and test metrics are close. Test R\u00b2 of ${closeTestR2.toFixed(3)} with only a ${closeR2Drop.toFixed(4)} drop from training confirms the model captures real patterns rather than memorising data. Test MAE is ${maeRatio.toFixed(1)}x the train MAE, within acceptable bounds.`;
  } else if (closeR2Drop < 0.15 && closeTestR2 > 0.7) {
    verdictIcon.textContent = "\u26A0";
    verdictTitle.textContent = "Acceptable Fit \u2014 Some generalisation gap";
    verdictTitle.style.color = "var(--moderate)";
    verdictText.textContent = `Test R\u00b2 of ${closeTestR2.toFixed(3)} is reasonable, but the ${closeR2Drop.toFixed(4)} drop from training and ${maeRatio.toFixed(1)}x MAE increase suggest mild overfitting. Predictions are useful for relative comparison but may have higher uncertainty for specific ranks.`;
  } else {
    verdictIcon.textContent = "\u274C";
    verdictTitle.textContent = "Overfitting Detected \u2014 Use with caution";
    verdictTitle.style.color = "var(--risky)";
    verdictText.textContent = `Significant gap between train and test performance (R\u00b2 drop: ${closeR2Drop.toFixed(4)}, MAE ratio: ${maeRatio.toFixed(1)}x). The model may be memorising training patterns. Predictions should be treated as rough estimates only.`;
  }

  // Scatter plot (test set)
  const scatterCtx = getCtx("chart-scatter");
  const scatterData = perf.scatter;

  const minVal = Math.min(...scatterData.map((d) => Math.min(d.actual, d.predicted)));
  const maxVal = Math.max(...scatterData.map((d) => Math.max(d.actual, d.predicted)));

  const scatterChart = new Chart(scatterCtx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Predicted vs Actual (Test Set)",
          data: scatterData.map((d) => ({ x: d.actual, y: d.predicted })),
          backgroundColor: ACCENT + "40",
          borderColor: ACCENT + "80",
          borderWidth: 0.5,
          pointRadius: 2.5,
          pointHoverRadius: 5,
        },
        {
          label: "Perfect Prediction",
          data: [
            { x: minVal, y: minVal },
            { x: maxVal, y: maxVal },
          ],
          type: "line",
          borderColor: "#e05a3a",
          borderWidth: 1.5,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: "Actual Closing Rank", color: "#9b978f" },
          grid: { color: "rgba(255,255,255,0.03)" },
        },
        y: {
          title: { display: true, text: "Predicted Closing Rank", color: "#9b978f" },
          grid: { color: "rgba(255,255,255,0.03)" },
        },
      },
      plugins: {
        legend: {
          labels: { filter: (item) => item.text !== "Perfect Prediction" },
        },
        tooltip: {
          backgroundColor: "rgba(14,17,24,0.95)",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
          callbacks: {
            label: (ctx) =>
              `Actual: ${ctx.raw.x.toLocaleString()}, Predicted: ${ctx.raw.y.toLocaleString()}`,
          },
        },
      },
    },
  });
  storeChart("chart-scatter", scatterChart);

  // Error histogram (test set)
  const histCtx = getCtx("chart-error-hist");
  const histData = perf.histogram;

  const histChart = new Chart(histCtx, {
    type: "bar",
    data: {
      labels: histData.map((d) => Math.round((d.bin_start + d.bin_end) / 2).toLocaleString()),
      datasets: [
        {
          label: "Frequency",
          data: histData.map((d) => d.count),
          backgroundColor: TEAL + "60",
          borderColor: TEAL,
          borderWidth: 1,
          borderRadius: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: "Prediction Error (Predicted \u2212 Actual)", color: "#9b978f" },
          grid: { display: false },
          ticks: {
            maxTicksLimit: 15,
            font: { size: 10 },
          },
        },
        y: {
          title: { display: true, text: "Frequency", color: "#9b978f" },
          grid: { color: "rgba(255,255,255,0.03)" },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(14,17,24,0.95)",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
        },
      },
    },
  });
  storeChart("chart-error-hist", histChart);
}


// \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 INIT \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
showPage("home");

