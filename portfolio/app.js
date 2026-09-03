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
        state.report.demo_mode = "sample fixture";
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
    var mode = byId("trace-mode");

    viewer.dataset.state = "ready";
    viewer.setAttribute("aria-busy", "false");
    run.textContent = report.run_id || "unnamed-run";
    status.textContent = trace.stop_reason || "ready";
    mode.textContent = (report.demo_mode || "research report") + " / inspectable trace";
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
    var verification = report.semantic_verification || report.verification || {};
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
      "<p class='stage-lede'>The default verifier exposes lexical support and review warnings. Optional evidence-only semantic verification adds calibrated verdicts when model access is configured.</p>" +
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

  function plainText(value) {
    var element = document.createElement("div");
    element.innerHTML = String(value || "");
    return (element.textContent || "").replace(/\s+/g, " ").trim();
  }

  function wikipediaSearch(question) {
    var url = new URL("https://en.wikipedia.org/w/api.php");
    url.search = new URLSearchParams({
      action: "query",
      list: "search",
      srsearch: question,
      srlimit: "3",
      format: "json",
      origin: "*"
    }).toString();
    return fetch(url.toString()).then(function (response) {
      if (!response.ok) throw new Error("Wikipedia returned " + response.status);
      return response.json();
    }).then(function (payload) {
      return arrayOrEmpty(payload && payload.query && payload.query.search).map(function (item, index) {
        return {
          id: "wikipedia-" + String(index + 1),
          provider: "wikipedia",
          title: item.title,
          url: "https://en.wikipedia.org/?curid=" + encodeURIComponent(item.pageid),
          snippet: plainText(item.snippet) || "Wikipedia search result for " + item.title + ".",
          kind: "encyclopedia",
          published_at: null,
          authors: [],
          credibility: "medium",
          evidence_text: plainText(item.snippet) || null,
          start_index: null,
          end_index: null,
          metadata: { page_id: item.pageid }
        };
      });
    });
  }

  function rebuildAbstract(index) {
    if (!index || typeof index !== "object") return "";
    var words = [];
    Object.keys(index).forEach(function (word) {
      arrayOrEmpty(index[word]).forEach(function (position) {
        words[position] = word;
      });
    });
    return words.join(" ").replace(/\s+/g, " ").trim();
  }

  function openAlexSearch(question) {
    var url = new URL("https://api.openalex.org/works");
    var safeQuestion = question.replace(/[?*]/g, " ").replace(/\s+/g, " ").trim();
    url.search = new URLSearchParams({ search: safeQuestion, "per-page": "3" }).toString();
    return fetch(url.toString()).then(function (response) {
      if (!response.ok) throw new Error("OpenAlex returned " + response.status);
      return response.json();
    }).then(function (payload) {
      return arrayOrEmpty(payload && payload.results).map(function (item, index) {
        var abstract = rebuildAbstract(item.abstract_inverted_index);
        var authors = arrayOrEmpty(item.authorships).slice(0, 6).map(function (entry) {
          return entry && entry.author ? entry.author.display_name : "";
        }).filter(Boolean);
        return {
          id: "openalex-" + String(index + 1),
          provider: "openalex",
          title: item.display_name || item.title || "Untitled work",
          url: (item.primary_location && item.primary_location.landing_page_url) || item.doi || item.id,
          snippet: abstract || ((item.display_name || "This work") + " was indexed by OpenAlex in " + (item.publication_year || "an unrecorded year") + "."),
          kind: "academic",
          published_at: item.publication_year ? String(item.publication_year) : null,
          authors: authors,
          credibility: "high",
          evidence_text: abstract || null,
          start_index: null,
          end_index: null,
          metadata: { cited_by_count: item.cited_by_count || 0, openalex_id: item.id }
        };
      });
    });
  }

  function publicReport(question, providerResults, elapsedMs) {
    var sources = [];
    var statuses = providerResults.map(function (result) {
      if (result.status === "fulfilled") {
        sources = sources.concat(result.value.sources);
        return { provider_id: result.value.provider, ok: true, result_count: result.value.sources.length, error: null };
      }
      return { provider_id: result.provider, ok: false, result_count: 0, error: result.reason && result.reason.message ? result.reason.message : "Request failed" };
    });
    if (!sources.length) throw new Error("Neither public provider returned a source record.");
    var findings = sources.slice(0, 4).map(function (source, index) {
      return {
        finding_id: "excerpt-" + String(index + 1),
        statement: source.snippet,
        importance: index < 2 ? "high" : "medium",
        confidence: source.provider === "openalex" ? 0.82 : 0.72,
        citation_ids: [source.id]
      };
    });
    var checks = findings.map(function (finding) {
      return {
        claim_id: finding.finding_id,
        statement: finding.statement,
        citation_ids: finding.citation_ids,
        verdict: "supported",
        coverage: 1,
        matched_evidence: [{ evidence_text: finding.statement }]
      };
    });
    var providerCount = statuses.filter(function (item) { return item.ok; }).length;
    return {
      run_id: "browser-" + Date.now().toString(36),
      question: question,
      executive_summary: "This browser run retrieved " + sources.length + " source records from " + providerCount + " public providers. Findings below are direct source excerpts, not model synthesis.",
      key_findings: findings,
      comparison: [{
        dimension: "Provider perspective",
        consensus: "The records offer encyclopedia and academic perspectives for direct review.",
        disagreements: ["Browser mode does not infer agreement or disagreement between source claims."],
        source_views: sources.slice(0, 4).map(function (source) { return { source_id: source.id, position: source.snippet }; })
      }],
      limitations: ["Public browser mode retrieves source records without model synthesis.", "Search ranking is controlled by each provider."],
      sources: sources,
      provider_status: statuses,
      audit: { citation_coverage: 1, grounding_score: 1, source_diversity: Math.min(1, providerCount / 2), comparison_quality: 0, score: 0.8, cited_finding_count: findings.length, finding_count: findings.length, provider_count: providerCount, unresolved_citations: [], warnings: ["Direct excerpts are not a semantic comparison."] },
      verification: { claim_checks: checks, supported_claim_count: checks.length, partial_claim_count: 0, unsupported_claim_count: 0, contradicted_claim_count: 0, claim_coverage: 1, average_coverage: 1, method: "direct_excerpt" },
      semantic_verification: null,
      model: "none / public browser retrieval",
      tool_calls: [{ name: "search_sources", query: question, source_count: sources.length, providers: statuses.filter(function (item) { return item.ok; }).map(function (item) { return item.provider_id; }), planned_queries: [question], covered_intents: ["public source discovery"], missing_intents: [], retrieval_coverage: providerCount / 2, stop_reason: providerCount === 2 ? "public_sources_retrieved" : "partial_provider_result" }],
      generated_at: new Date().toISOString(),
      demo_mode: "live public retrieval / " + elapsedMs + " ms"
    };
  }

  function runPublicResearch(question) {
    var submit = byId("research-submit");
    var status = byId("research-status");
    var started = performance.now();
    submit.disabled = true;
    submit.textContent = "Searching sources...";
    status.textContent = "Querying Wikipedia and OpenAlex in parallel.";
    setViewerState("loading", "Searching public sources.", "Waiting for Wikipedia and OpenAlex...", "·");
    var requests = [
      wikipediaSearch(question).then(function (sources) { return { provider: "wikipedia", sources: sources }; }),
      openAlexSearch(question).then(function (sources) { return { provider: "openalex", sources: sources }; })
    ];
    return Promise.all(requests.map(function (promise, index) {
      var provider = index === 0 ? "wikipedia" : "openalex";
      return promise.then(function (value) { return { status: "fulfilled", value: value, provider: provider }; })
        .catch(function (reason) { return { status: "rejected", reason: reason, provider: provider }; });
    })).then(function (results) {
      state.report = publicReport(question, results, Math.round(performance.now() - started));
      state.step = 0;
      state.selectedEvidenceId = null;
      renderReport();
      status.textContent = "Live run complete. " + state.report.sources.length + " source records are ready to inspect.";
    }).catch(function (error) {
      setViewerState("error", "Public search could not complete.", error.message || "The providers did not return usable records.", "!");
      status.textContent = "Search failed. The deterministic fixture is still available below.";
    }).finally(function () {
      submit.disabled = false;
      submit.innerHTML = "Run public search <span aria-hidden='true'>↘</span>";
    });
  }

  function init() {
    byId("research-form").addEventListener("submit", function (event) {
      event.preventDefault();
      var question = byId("research-question").value.trim();
      if (question.length >= 8) runPublicResearch(question);
    });
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
