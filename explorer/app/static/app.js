// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

(() => {
  const themeToggle = document.querySelector("[data-theme-toggle]");
  const form = document.querySelector(".editor form");
  const results = document.querySelector("#results");
  installThemeToggle();
  installSampleDropdowns();
  installLanguageGuideModal();
  highlightLanguageGuideCards();
  installQueryHighlighting();
  if (!results) return;

  highlightResults();
  installCodeBlockControls();
  installBenchmarkCompileModal();
  if (!form || !window.fetch) return;

  const archSelect = form.querySelector('select[name="arch"]');
  const targetSelect = form.querySelector('select[name="benchmark_target"]');
  const inputModeSelect = form.querySelector('select[name="input_mode"]');
  const benchmarkButton = form.querySelector('button[data-role="benchmark"]');
  const copyQueryButton = form.querySelector('button[data-role="copy-query"]');
  const shareButton = form.querySelector('button[data-role="share-link"]');
  const availabilityMatrix = readBenchmarkAvailabilityMatrix();
  const urlOptionFields = new Set([
    "arch",
    "benchmark_target",
    "provider",
    "rows",
    "warmups",
    "runs",
    "null_density",
    "output",
    "input_mode",
  ]);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const submitter = event.submitter;
    if (submitter?.disabled) return;
    const action = submitter?.getAttribute("formaction") || form.action || "/compile";
    await submitFragment(action, submitter);
  });
  form.addEventListener("change", (event) => {
    const control = event.target;
    if (!control?.name || !urlOptionFields.has(control.name)) return;
    syncCurrentOptionsToUrl();
  });

  async function submitFragment(action, submitter = null) {
    const buttons = Array.from(form.querySelectorAll("button"));
    const previousLabel = submitter?.textContent;

    results.classList.add("loading");
    form.classList.add("submitting");
    if (submitter) submitter.textContent = "Working...";
    buttons.forEach((button) => {
      button.disabled = true;
    });

    try {
      const response = await fetch(action, {
        method: "POST",
        body: new FormData(form),
        headers: { "X-Simjit-Explorer-Fragment": "results" },
      });
      const html = await response.text();
      if (!response.ok) throw new Error(html || response.statusText);
      results.innerHTML = html;
      highlightResults();
    } catch (error) {
      results.innerHTML = `<div class="diagnostic error">${escapeHtml(error.message || String(error))}</div>`;
    } finally {
      results.classList.remove("loading");
      form.classList.remove("submitting");
      if (submitter && previousLabel) submitter.textContent = previousLabel;
      buttons.forEach((button) => {
        button.disabled = false;
      });
      refreshBenchmarkAvailability();
    }
  }

  if (archSelect) {
    archSelect.addEventListener("change", () => {
      refreshBenchmarkAvailability();
      void submitFragment("compile");
    });
    refreshBenchmarkAvailability();
  }
  if (targetSelect) {
    targetSelect.addEventListener("change", refreshBenchmarkAvailability);
    refreshBenchmarkAvailability();
  }
  if (inputModeSelect) {
    inputModeSelect.addEventListener("change", refreshBenchmarkAvailability);
    refreshBenchmarkAvailability();
  }
  if (copyQueryButton) {
    copyQueryButton.addEventListener("click", () => {
      void copyQueryText(copyQueryButton);
    });
  }
  if (shareButton) {
    shareButton.addEventListener("click", () => {
      void copyShareLink(shareButton);
    });
  }
  syncCurrentOptionsToUrl();

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function installThemeToggle() {
    if (window.simjitTheme) {
      window.simjitTheme.bindThemeToggle(themeToggle);
      return;
    }
    document.documentElement.dataset.theme = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  }

  function installSampleDropdowns() {
    const sampleNav = document.querySelector(".sample-nav");
    if (!sampleNav) return;

    sampleNav.addEventListener("toggle", (event) => {
      const opened = event.target.closest(".sample-group");
      if (!opened || !opened.open) return;
      sampleNav.querySelectorAll(".sample-group[open]").forEach((group) => {
        if (group !== opened) group.removeAttribute("open");
      });
    }, true);

    sampleNav.addEventListener("click", (event) => {
      const link = event.target.closest(".sample-links a");
      if (!link) return;
      preserveCurrentOptions(link);
      sampleNav.querySelectorAll(".sample-group[open]").forEach((group) => {
        group.removeAttribute("open");
      });
    });

    document.addEventListener("pointerdown", (event) => {
      if (sampleNav.contains(event.target)) return;
      closeSampleDropdowns();
    });

    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      closeSampleDropdowns();
    });

    function closeSampleDropdowns() {
      sampleNav.querySelectorAll(".sample-group[open]").forEach((group) => {
        group.removeAttribute("open");
      });
    }
  }

  function installLanguageGuideModal() {
    const openButton = document.querySelector("[data-language-guide-open]");
    const modal = document.querySelector("[data-language-guide-modal]");
    const closeButton = modal?.querySelector("[data-language-guide-close]");
    if (!openButton || !modal) return;

    let returnFocus = null;

    openButton.addEventListener("click", () => {
      returnFocus = openButton;
      if (typeof modal.showModal === "function") {
        modal.showModal();
      } else {
        modal.setAttribute("open", "");
      }
      closeButton?.focus();
    });

    closeButton?.addEventListener("click", () => {
      closeLanguageGuide();
    });

    modal.addEventListener("click", (event) => {
      if (event.target === modal) closeLanguageGuide();
    });

    modal.addEventListener("close", restoreGuideFocus);

    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape" || !modal.hasAttribute("open")) return;
      if (typeof modal.close === "function") return;
      closeLanguageGuide();
    });

    function closeLanguageGuide() {
      if (typeof modal.close === "function") {
        modal.close();
        return;
      }
      modal.removeAttribute("open");
      restoreGuideFocus();
    }

    function restoreGuideFocus() {
      returnFocus?.focus();
      returnFocus = null;
    }
  }

  function installBenchmarkCompileModal() {
    results.addEventListener("click", (event) => {
      const openButton = event.target.closest("[data-benchmark-compile-open]");
      if (openButton) {
        event.preventDefault();
        event.stopPropagation();
        const modal = openButton.closest(".pane")?.querySelector("[data-benchmark-compile-modal]");
        if (!modal) return;
        modal._simjitReturnFocus = openButton;
        modal.addEventListener("close", restoreBenchmarkModalFocus, { once: true });
        if (typeof modal.showModal === "function") {
          modal.showModal();
        } else {
          modal.setAttribute("open", "");
        }
        modal.querySelector("[data-benchmark-compile-close]")?.focus();
        return;
      }

      const closeButton = event.target.closest("[data-benchmark-compile-close]");
      if (closeButton) {
        event.preventDefault();
        event.stopPropagation();
        closeBenchmarkCompileModal(closeButton.closest("[data-benchmark-compile-modal]"));
        return;
      }

      const modal = event.target.closest("[data-benchmark-compile-modal]");
      if (modal && event.target === modal) {
        closeBenchmarkCompileModal(modal);
      }
    });

    results.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      const modal = results.querySelector("[data-benchmark-compile-modal][open]");
      if (!modal || typeof modal.close === "function") return;
      event.preventDefault();
      closeBenchmarkCompileModal(modal);
    });

    function closeBenchmarkCompileModal(modal) {
      if (!modal) return;
      if (typeof modal.close === "function") {
        modal.close();
        return;
      }
      modal.removeAttribute("open");
      restoreBenchmarkModalFocus.call(modal);
    }

    function restoreBenchmarkModalFocus() {
      const target = this._simjitReturnFocus;
      this._simjitReturnFocus = null;
      if (target instanceof HTMLElement) target.focus();
    }
  }

  function installQueryHighlighting() {
    const editor = document.querySelector("[data-query-editor]");
    const textarea = editor?.querySelector("textarea");
    const highlight = editor?.querySelector("[data-query-highlight]");
    const highlightPre = highlight?.closest("pre");
    const modeSelect = form?.querySelector('select[name="input_mode"]');
    if (!editor || !textarea || !highlight || !highlightPre) return;
    let activeMode = modeSelect?.value || "expression_sql";

    const syncHighlight = () => {
      const text = textarea.value;
      const language = activeMode === "serialized_hir" ? "lisp" : "sql";
      if (window.hljs?.getLanguage?.(language)) {
        highlight.innerHTML = window.hljs.highlight(text, {
          language,
          ignoreIllegals: true,
        }).value;
      } else {
        highlight.textContent = text;
      }
      if (text.endsWith("\n")) highlight.append(document.createTextNode(" "));
      syncQueryScroll();
    };

    const syncQueryScroll = () => {
      highlightPre.scrollTop = textarea.scrollTop;
      highlightPre.scrollLeft = textarea.scrollLeft;
    };

    const resetEditorForMode = () => {
      const nextMode = modeSelect?.value || "expression_sql";
      if (nextMode === activeMode) {
        syncHighlight();
        return;
      }
      activeMode = nextMode;
      textarea.value = "";
      textarea.scrollTop = 0;
      textarea.scrollLeft = 0;
      syncHighlight();
      textarea.focus();
    };

    textarea.addEventListener("input", syncHighlight);
    textarea.addEventListener("scroll", syncQueryScroll);
    textarea.addEventListener("change", syncHighlight);
    modeSelect?.addEventListener("change", resetEditorForMode);
    syncHighlight();
  }

  function refreshBenchmarkAvailability() {
    if (!archSelect || !benchmarkButton) return;
    const inputMode = inputModeSelect?.value || "expression_sql";
    const targetId = targetSelect?.value || "local";
    const archValue = archSelect.value || "native";
    const availability = availabilityFor(inputMode, targetId, archValue);
    const runnable = Boolean(availability?.runnable);
    benchmarkButton.disabled = !runnable;
    benchmarkButton.title = runnable
      ? ""
      : availability?.title || availability?.reason || "Benchmark target is unavailable";
  }

  function readBenchmarkAvailabilityMatrix() {
    const node = document.querySelector("#benchmark-availability-data");
    if (!node?.textContent) return {};
    try {
      return JSON.parse(node.textContent);
    } catch (error) {
      return {};
    }
  }

  function availabilityFor(inputMode, targetId, arch) {
    return availabilityMatrix?.[inputMode]?.[targetId]?.[arch]
      || availabilityMatrix?.[inputMode]?.[targetId]?.native
      || null;
  }

  async function copyShareLink(button) {
    const textarea = form.querySelector('textarea[name="query"]');
    if (!textarea) return;
    const original = button.textContent;
    try {
      const url = new URL(window.location.pathname || "/", window.location.origin);
      url.searchParams.set("query", base64UrlEncodeText(textarea.value));
      appendCurrentOptions(url, { includeInputMode: true });
      await copyText(url.toString());
      button.textContent = "Link copied";
    } catch (error) {
      button.textContent = "Copy failed";
    } finally {
      setTimeout(() => {
        button.textContent = original;
      }, 1200);
    }
  }

  async function copyQueryText(button) {
    const textarea = form.querySelector('textarea[name="query"]');
    if (!textarea) return;
    const original = button.textContent;
    try {
      await copyText(textarea.value);
      button.textContent = "Query copied";
    } catch (error) {
      button.textContent = "Copy failed";
    } finally {
      setTimeout(() => {
        button.textContent = original;
      }, 1200);
    }
  }

  function preserveCurrentOptions(link) {
    if (!form) return;
    const url = new URL(link.href, window.location.origin);
    appendCurrentOptions(url);
    link.href = `${url.search}${url.hash}`;
  }

  function syncCurrentOptionsToUrl() {
    if (!window.history?.replaceState) return;
    const url = new URL(window.location.href);
    appendCurrentOptions(url, { includeInputMode: true });
    const next = `${url.pathname}${url.search}${url.hash}`;
    const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (next !== current) window.history.replaceState(null, "", next);
  }

  function appendCurrentOptions(url, { includeInputMode = false } = {}) {
    if (!form) return;
    [
      "benchmark_target",
      "provider",
      "rows",
      "warmups",
      "runs",
      "null_density",
      "output",
      ...(includeInputMode ? ["input_mode"] : []),
    ].forEach((name) => {
      const control = form.elements[name];
      if (!control || control.disabled) return;
      setOptionalSearchParam(url, name, control);
    });
    appendArchOption(url);
  }

  function setOptionalSearchParam(url, name, control) {
    const value = control.value;
    if (isDefaultControlValue(control, value)) {
      url.searchParams.delete(name);
      return;
    }
    url.searchParams.set(name, value);
  }

  function isDefaultControlValue(control, value) {
    const defaultValue = control.dataset.defaultValue;
    if (defaultValue === undefined) return false;
    if (control.type === "number") {
      const numericValue = Number(value);
      const numericDefault = Number(defaultValue);
      if (Number.isFinite(numericValue) && Number.isFinite(numericDefault)) {
        return numericValue === numericDefault;
      }
    }
    return value === defaultValue;
  }

  function appendArchOption(url) {
    const control = form.elements.arch;
    if (!control || control.disabled) return;
    const archValue = control.value;
    const nativeArch = currentTargetNativeArch();
    if (nativeArch && archValue === nativeArch) {
      url.searchParams.delete("arch");
      return;
    }
    url.searchParams.set("arch", archValue);
  }

  function currentTargetNativeArch() {
    const inputMode = inputModeSelect?.value || "expression_sql";
    const targetId = targetSelect?.value || "local";
    const nativeAvailability = availabilityFor(inputMode, targetId, "native")
      || availabilityFor("expression_sql", targetId, "native");
    return nativeAvailability?.native_arch || "";
  }

  function base64UrlEncodeText(text) {
    const bytes = new TextEncoder().encode(text);
    let binary = "";
    const chunkSize = 0x8000;
    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
      const chunk = bytes.subarray(offset, offset + chunkSize);
      binary += String.fromCharCode(...chunk);
    }
    return btoa(binary)
      .replaceAll("+", "-")
      .replaceAll("/", "_")
      .replace(/=+$/, "");
  }

  function highlightResults() {
    highlightCodeBlocks(results.querySelectorAll("pre code"));
  }

  function highlightLanguageGuideCards() {
    highlightCodeBlocks(document.querySelectorAll(".language-card pre code"), "sql");
  }

  function highlightCodeBlocks(blocks, language = "") {
    if (!window.hljs) return;
    blocks.forEach((block) => {
      if (language) {
        block.classList.add(`language-${language}`);
        block.closest("pre")?.classList.add(`language-${language}`);
      }
      block.removeAttribute("data-highlighted");
      window.hljs.highlightElement(block);
    });
  }

  function installCodeBlockControls() {
    results.addEventListener("click", (event) => {
      const copyButton = event.target.closest("[data-copy-code]");
      if (copyButton) {
        event.preventDefault();
        event.stopPropagation();
        copyCodeBlock(copyButton);
        return;
      }

      const copyCsvButton = event.target.closest("[data-copy-benchmark-csv]");
      if (copyCsvButton) {
        event.preventDefault();
        event.stopPropagation();
        copyBenchmarkCsv(copyCsvButton);
        return;
      }

      const infoTooltip = event.target.closest("[data-tooltip]");
      if (infoTooltip) {
        event.preventDefault();
        event.stopPropagation();
        return;
      }

      const code = event.target.closest("pre code");
      if (code) code.focus({ preventScroll: true });
    });

    results.addEventListener("keydown", (event) => {
      const code = event.target.closest("pre code");
      if (!code || event.key.toLowerCase() !== "a" || !(event.metaKey || event.ctrlKey)) return;
      event.preventDefault();
      selectCodeBlock(code);
    });
  }

  async function copyCodeBlock(button) {
    const code = button.closest(".pane")?.querySelector("pre code");
    if (!code) return;
    const text = code.textContent || "";
    const original = button.textContent;
    try {
      await copyText(text);
      button.textContent = "Copied";
    } catch (error) {
      button.textContent = "Copy failed";
    } finally {
      setTimeout(() => {
        button.textContent = original;
      }, 1200);
    }
  }

  async function copyBenchmarkCsv(button) {
    const table = button.closest(".pane")?.querySelector(".bench-table");
    if (!table) return;
    const original = button.textContent;
    try {
      await copyText(tableToCsv(table));
      button.textContent = "Copied";
    } catch (error) {
      button.textContent = "Copy failed";
    } finally {
      setTimeout(() => {
        button.textContent = original;
      }, 1200);
    }
  }

  function tableToCsv(table) {
    return Array.from(table.querySelectorAll("tr"))
      .map((row) => (
        Array.from(row.children)
          .map((cell) => csvCell(cell.textContent || ""))
          .join(",")
      ))
      .join("\n");
  }

  function csvCell(value) {
    const normalized = String(value).replace(/\s+/g, " ").trim();
    if (!/[",\n\r]/.test(normalized)) return normalized;
    return `"${normalized.replaceAll('"', '""')}"`;
  }

  async function copyText(text) {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }

    const scratch = document.createElement("textarea");
    scratch.value = text;
    scratch.setAttribute("readonly", "");
    scratch.style.position = "fixed";
    scratch.style.opacity = "0";
    document.body.appendChild(scratch);
    scratch.select();
    document.execCommand("copy");
    scratch.remove();
  }

  function selectCodeBlock(code) {
    const selection = window.getSelection();
    if (!selection) return;
    const range = document.createRange();
    range.selectNodeContents(code);
    selection.removeAllRanges();
    selection.addRange(range);
  }
})();
