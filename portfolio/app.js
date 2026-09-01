(function () {
  "use strict";

  var steps = [
    "Question",
    "Planned queries",
    "Providers",
    "Evidence",
    "Claims",
    "Verification"
  ];
  var state = {
    report: null,
    step: 0,
    selectedEvidenceId: null
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function arrayOrEmpty(value) {
    return Array.isArray(value) ? value : [];
  }

  function escapeHTML(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function percentage(value) {
    return typeof value === "number" && isFinite(value)
      ? Math.round(value * 100) + "%"
      : "n/a";
  }

  function traceOf(report) {
    return arrayOrEmpty(report && report.tool_calls)[0] || {};
  }

  function sourceById(report, id) {
    return arrayOrEmpty(report && report.sources).filter(function (source) {
      return source && source.id === id;
    })[0] || null;
  }

  function safeSourceURL(value) {
    if (typeof value !== "string" || !value.trim()) {
      return "#";
    }
    try {
      var parsed = new URL(value, window.location.href);
      return parsed.protocol === "http:" || parsed.protocol === "https:"
        ? parsed.href
        : "#";
    } catch (error) {
      return "#";
    }
  }

  function setViewerState(kind, title, detail, icon) {
    var viewer = byId("trace-viewer");
    var stage = byId("trace-stage");
    var retry = byId("trace-retry");
    var previous = byId("trace-prev");
    var next = byId("trace-next");
    var run = byId("trace-run");
    var status = byId("trace-status");
    var progress = byId("trace-progress");
    var meter = byId("trace-meter-fill");

    viewer.dataset.state = kind;
    viewer.setAttribute("aria-busy", kind === "loading" ? "true" : "false");
    run.textContent = kind === "loading" ? "loading sample..." : "trace unavailable";
    status.textContent = kind;
    progress.textContent = "-- / --";
    meter.style.width = "0";
    previous.disabled = true;
    next.disabled = true;
    retry.classList.toggle("is-hidden", kind === "loading");

    var action = kind === "loading"
      ? ""
      : "<button class='state-action' type='button' data-action='retry'>Retry fixture</button>";
    stage.innerHTML =
      "<div class='viewer-state viewer-" + escapeHTML(kind) + "'>" +
        "<span class='state-icon' aria-hidden='true'>" + escapeHTML(icon || "!") + "</span>" +
        "<p><strong>" + escapeHTML(title) + "</strong><br>" + escapeHTML(detail) + "</p>" +
        action +
      "</div>";
  }

  function loadReport() {
    state.report = null;
    state.step = 0;
    state.selectedEvidenceId = null;
    setViewerState(
      "loading",
      "Loading the evidence chain.",
      "Reading the deterministic sample report...",
      "·"
    );

    fetch("sample_report.json", { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("sample_report.json returned " + response.status);
        }
        return response.json();
      })
      .then(function (report) {
        if (!report || typeof report !== "object" || Object.keys(report).length === 0) {
          setViewerState(
            "empty",
            "No evidence chain is loaded.",
            "The viewer is ready, but the report payload is empty.",
            "∅"
          );
          return;
        }
        if (!report.question || !Array.isArray(report.sources)) {
          throw new Error("The report is missing its question or source records.");
        }
        state.report = report;
        renderReport();
      })
      .catch(function (error) {
        setViewerState(
          "error",
          "The evidence chain could not be loaded.",
          error && error.message
            ? error.message + " Serve the portfolio directory over HTTP and retry."
            : "Serve the portfolio directory over HTTP and retry.",
          "!"
        );
      });
  }

  function renderReport() {
    var report = state.report;
    var trace = traceOf(report);
    var viewer = byId("trace-viewer");
    var run = byId("trace-run");
    var status = byId("trace-status");
    var retry = byId("trace-retry");

    viewer.dataset.state = "ready";
    viewer.setAttribute("aria-busy", "false");
    run.textContent = report.run_id || "unnamed-run";
    status.textContent = trace.stop_reason || "ready";
    retry.classList.add("is-hidden");
    renderNavigation();
    renderStep();
  }

  function renderNavigation() {
    var nav = byId("trace-nav");
    var buttons = nav.querySelectorAll(".trace-step");
    var progress = byId("trace-progress");
    var meter = byId("trace-meter-fill");
    var previous = byId("trace-prev");
    var next = byId("trace-next");

    Array.prototype.forEach.call(buttons, function (button, index) {
      var active = index === state.step;
      button.classList.toggle("is-current", active);
      if (active) {
        button.setAttribute("aria-current", "step");
      } else {
        button.removeAttribute("aria-current");
      }
    });

    progress.textContent = String(state.step + 1).padStart(2, "0") + " / " + String(steps.length).padStart(2, "0");
    meter.style.width = ((state.step + 1) / steps.length * 100) + "%";
    previous.disabled = state.step === 0;
    next.disabled = state.step === steps.length - 1;
    next.innerHTML = state.step === steps.length - 1
      ? "Trace complete <span aria-hidden='true'>✓</span>"
      : "Next stage <span aria-hidden='true'>→</span>";
  }

  function renderStep() {
    var report = state.report;
    var renderers = [
      renderQuestion,
      renderQueries,
      renderProviders,
      renderEvidence,
      renderClaims,
      renderVerification
    ];
    byId("trace-stage").innerHTML = renderers[state.step](report);
    renderNavigation();
  }

  function renderQuestion(report) {
    var trace = traceOf(report);
    var sources = arrayOrEmpty(report.sources);
    var findings = arrayOrEmpty(report.key_findings);
    return (
      "<div class='stage-kicker'>01 / QUESTION</div>" +
      "<h3 class='stage-title'>Start with a question, not a prompt-shaped answer.</h3>" +
      "<p class='stage-lede'>The agent preserves the original question as the anchor for every later trace decision.</p>" +
      "<div class='question-card'><p>" + escapeHTML(report.question) + "</p></div>" +
      "<div class='stage-grid'>" +
        metric("source records", sources.length) +
        metric("draft claims", findings.length) +
        metric("retrieval coverage", percentage(trace.retrieval_coverage)) +
      "</div>"
    );
  }

  function renderQueries(report) {
    var trace = traceOf(report);
    var queries = arrayOrEmpty(trace.planned_queries);
    var queryMarkup = queries.length
      ? queries.map(function (query, index) {
          return (
            "<li class='query-item'>" +
              "<span>" + String(index + 1).padStart(2, "0") + "</span>" +
              "<code>" + escapeHTML(query) + "</code>" +
            "</li>"
          );
        }).join("")
      : "<li class='query-item'><span>n/a</span><code>No planned queries were recorded.</code></li>";
    return (
      "<div class='stage-kicker'>02 / PLANNED QUERIES</div>" +
      "<h3 class='stage-title'>Decompose the question before retrieval fans out.</h3>" +
      "<p class='stage-lede'>Each query is bounded, purpose-built, and visible. The executor stops when coverage is good enough for the trace.</p>" +
      "<ol class='query-list'>" + queryMarkup + "</ol>" +
      "<div class='stop-banner'>" +
        "<span class='data-label'>Stop decision</span>" +
        "<strong>" + escapeHTML(trace.stop_reason || "not recorded") + "</strong>" +
        "<small>coverage " + percentage(trace.retrieval_coverage) + " · covered " +
          escapeHTML(arrayOrEmpty(trace.covered_intents).join(", ") || "none") + "</small>" +
      "</div>"
    );
  }

  function renderProviders(report) {
    var providers = arrayOrEmpty(report.provider_status);
    var providerMarkup = providers.length
      ? providers.map(function (provider) {
          var ok = provider.ok !== false;
          return (
            "<li class='provider-card " + (ok ? "" : "is-error") + "'>" +
              "<div><strong>" + escapeHTML(provider.provider_id || "unknown provider") + "</strong>" +
              "<small>" + (ok ? "returned " : "failed with ") +
                escapeHTML(provider.result_count === undefined ? "no count" : provider.result_count + " source records") +
                (provider.error ? " · " + escapeHTML(provider.error) : "") + "</small></div>" +
              "<span class='provider-badge'>" + (ok ? "healthy" : "error") + "</span>" +
            "</li>"
          );
        }).join("")
      : "<li class='provider-card'><div><strong>No provider status</strong><small>The report did not include provider health.</small></div></li>";
    return (
      "<div class='stage-kicker'>03 / PROVIDERS</div>" +
      "<h3 class='stage-title'>Independent sources make agreement inspectable.</h3>" +
      "<p class='stage-lede'>Provider health stays in the application-owned envelope, so a fluent report cannot hide a failed source.</p>" +
      "<ul class='provider-list'>" + providerMarkup + "</ul>" +
      "<div class='stage-grid'>" +
        metric("providers seen", providers.length) +
        metric("successful", providers.filter(function (item) { return item.ok !== false; }).length) +
        metric("source records", arrayOrEmpty(report.sources).length) +
      "</div>"
    );
  }

  function renderEvidence(report) {
    var sources = arrayOrEmpty(report.sources);
    var evidenceMarkup = sources.length
      ? sources.map(function (source) {
          var selected = state.selectedEvidenceId === source.id;
          var evidence = source.evidence_text || source.snippet || "No source-specific passage recorded.";
          var span = typeof source.start_index === "number" && typeof source.end_index === "number"
            ? "span " + source.start_index + " → " + source.end_index
            : "source-specific excerpt";
          return (
            "<article class='evidence-card " + (selected ? "is-selected" : "") + "' data-source-id='" + escapeHTML(source.id) + "'>" +
              "<div class='source-meta'><span class='source-kind'>" + escapeHTML(source.kind || "web") + "</span>" +
                "<span>·</span><span>" + escapeHTML(source.provider || "unknown") + "</span></div>" +
              "<h4>" + escapeHTML(source.title || "Untitled source") + "</h4>" +
              "<p>" + escapeHTML(evidence) + "</p>" +
              "<footer><span>" + escapeHTML(span) + "</span>" +
                "<a class='source-link' href='" + escapeHTML(safeSourceURL(source.url)) + "' target='_blank' rel='noreferrer'>Open source ↗</a></footer>" +
            "</article>"
          );
        }).join("")
      : "<div class='viewer-state viewer-empty'><span class='state-icon' aria-hidden='true'>∅</span><p><strong>No evidence passages.</strong><br>The report contains no source records to inspect.</p></div>";
    return (
      "<div class='stage-kicker'>04 / EVIDENCE PASSAGES</div>" +
      "<h3 class='stage-title'>Read the passage before trusting the claim.</h3>" +
      "<p class='stage-lede'>The viewer keeps a source-specific excerpt, provider identity, URL, and hosted-response offsets together.</p>" +
      "<div class='evidence-list'>" + evidenceMarkup + "</div>"
    );
  }

  function renderClaims(report) {
    var findings = arrayOrEmpty(report.key_findings);
    var claimMarkup = findings.length
      ? findings.map(function (finding) {
          var citations = arrayOrEmpty(finding.citation_ids);
          var citationMarkup = citations.map(function (citation) {
            return "<button class='citation-link' type='button' data-evidence-id='" + escapeHTML(citation) + "'>[" + escapeHTML(citation) + "]</button>";
          }).join("");
          return (
            "<article class='claim-card'>" +
              "<div class='claim-meta'><span>" + escapeHTML(finding.finding_id || "claim") + "</span><span>·</span><span>" +
                escapeHTML(finding.importance || "unrated") + "</span><span>·</span><span>" +
                percentage(finding.confidence) + " confidence</span></div>" +
              "<h4>Claim under review</h4>" +
              "<p>" + escapeHTML(finding.statement || "No claim statement recorded.") + "</p>" +
              "<div class='citation-row'>" + citationMarkup + "</div>" +
            "</article>"
          );
        }).join("")
      : "<div class='viewer-state viewer-empty'><span class='state-icon' aria-hidden='true'>∅</span><p><strong>No claims were synthesized.</strong><br>The report contains no findings to inspect.</p></div>";
    return (
      "<div class='stage-kicker'>05 / CLAIMS</div>" +
      "<h3 class='stage-title'>A conclusion is a claim with a citation set.</h3>" +
      "<p class='stage-lede'>Citation IDs remain clickable, so the reviewer can jump back from synthesis to the evidence passage that gave it shape.</p>" +
      "<div class='claim-list'>" + claimMarkup + "</div>"
    );
  }

  function renderVerification(report) {
    var verification = report.verification || {};
    var checks = arrayOrEmpty(verification.claim_checks);
    var audit = report.audit || {};
    var checkMarkup = checks.length
      ? checks.map(function (check) {
          var verdict = String(check.verdict || "unverified").toLowerCase();
          var match = arrayOrEmpty(check.matched_evidence)[0];
          return (
            "<article class='verification-card is-" + escapeHTML(verdict) + "'>" +
              "<div class='verification-meta'><span class='verdict'>" + escapeHTML(verdict) + "</span>" +
                "<span class='score'>coverage " + percentage(check.coverage) + "</span></div>" +
              "<h4>" + escapeHTML(check.claim_id || "claim") + "</h4>" +
              "<p>" + escapeHTML(check.statement || "No claim statement recorded.") + "</p>" +
              (match ? "<p><span class='data-label'>Matched passage</span>" + escapeHTML(match.evidence_text) + "</p>" : "") +
            "</article>"
          );
        }).join("")
      : "<div class='viewer-state viewer-empty'><span class='state-icon' aria-hidden='true'>∅</span><p><strong>No verification attached.</strong><br>This report has no claim-level checks.</p></div>";
    return (
      "<div class='stage-kicker'>06 / VERIFICATION</div>" +
      "<h3 class='stage-title'>Make uncertainty visible at the end of the trail.</h3>" +
      "<p class='stage-lede'>The deterministic verifier exposes support, partial overlap, and review warnings. It is deliberately labeled as lexical, not semantic entailment.</p>" +
      "<div class='audit-strip'>" +
        metric("grounding", percentage(audit.grounding_score)) +
        metric("citation coverage", percentage(audit.citation_coverage)) +
        metric("claim coverage", percentage(verification.claim_coverage)) +
        metric("method", verification.method || "n/a") +
      "</div>" +
      "<div class='verification-list'>" + checkMarkup + "</div>"
    );
  }

  function metric(label, value) {
    return "<div class='metric-block'><span>" + escapeHTML(label) + "</span><strong>" + escapeHTML(value) + "</strong></div>";
  }

  function goToStep(step) {
    if (!state.report) {
      return;
    }
    state.step = Math.max(0, Math.min(steps.length - 1, step));
    renderStep();
  }

  function init() {
    byId("trace-nav").addEventListener("click", function (event) {
      var button = event.target.closest(".trace-step");
      if (button) {
        goToStep(Number(button.dataset.step));
      }
    });
    byId("trace-prev").addEventListener("click", function () {
      goToStep(state.step - 1);
    });
    byId("trace-next").addEventListener("click", function () {
      goToStep(state.step + 1);
    });
    byId("trace-retry").addEventListener("click", loadReport);
    byId("trace-stage").addEventListener("click", function (event) {
      var retry = event.target.closest("[data-action='retry']");
      if (retry) {
        loadReport();
        return;
      }
      var evidenceLink = event.target.closest("[data-evidence-id]");
      if (evidenceLink) {
        state.selectedEvidenceId = evidenceLink.dataset.evidenceId;
        goToStep(3);
      }
    });
    loadReport();
  }

  window.SourceboundTrace = {
    load: loadReport,
    goToStep: goToStep
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}());
