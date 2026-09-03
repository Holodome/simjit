// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

(() => {
  const themeToggle = document.querySelector("[data-theme-toggle]");

  installThemeToggle();
  installSourceModal();
  installPythonBaselineToggle();
  highlightCode();

  function installThemeToggle() {
    if (window.simjitTheme) {
      window.simjitTheme.bindThemeToggle(themeToggle);
      return;
    }
    document.documentElement.dataset.theme = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  }

  function highlightCode() {
    if (!window.hljs) return;
    document.querySelectorAll("pre code").forEach((block) => {
      block.removeAttribute("data-highlighted");
      window.hljs.highlightElement(block);
    });
  }

  function installPythonBaselineToggle() {
    const toggle = document.querySelector("[data-python-baseline-toggle]");
    if (!toggle) return;

    const buttons = Array.from(toggle.querySelectorAll("[data-python-baseline]"));
    const cells = Array.from(document.querySelectorAll("[data-python-heat-cell]"));
    const baselineAxis = document.querySelector("[data-python-baseline-axis]");
    const initial =
      buttons.find((button) => button.getAttribute("aria-pressed") === "true")
        ?.dataset.pythonBaseline || buttons[0]?.dataset.pythonBaseline || "numba";

    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        applyBaseline(button.dataset.pythonBaseline || "numba");
      });
    });
    applyBaseline(initial);

    function applyBaseline(baseline) {
      buttons.forEach((button) => {
        button.setAttribute(
          "aria-pressed",
          button.dataset.pythonBaseline === baseline ? "true" : "false"
        );
      });
      const activeButton = buttons.find((button) => button.dataset.pythonBaseline === baseline);
      if (baselineAxis && activeButton?.dataset.pythonBaselineLabel) {
        baselineAxis.textContent = activeButton.dataset.pythonBaselineLabel;
      }

      cells.forEach((cell) => {
        const speedClass = cell.dataset[`${baseline}SpeedClass`];
        const speedupLabel = cell.dataset[`${baseline}SpeedupLabel`];
        const timeLabel = cell.dataset[`${baseline}TimeLabel`];
        if (!speedClass || !speedupLabel || !timeLabel) return;

        Array.from(cell.classList)
          .filter((className) => /^speed-\d+$/.test(className))
          .forEach((className) => cell.classList.remove(className));
        cell.classList.add(`speed-${speedClass}`);
        cell.dataset.currentPythonBaseline = baseline;
        const speedup = cell.querySelector("[data-heat-speedup]");
        const time = cell.querySelector("[data-heat-time]");
        if (speedup) speedup.textContent = speedupLabel;
        if (time) time.textContent = timeLabel;
      });
    }
  }

  function installSourceModal() {
    const modal = document.querySelector("[data-source-modal]");
    if (!modal) return;

    const title = modal.querySelector("[data-source-title]");
    const body = modal.querySelector("[data-source-body]");
    const closeButton = modal.querySelector("[data-source-close]");
    let returnFocus = null;

    document.querySelectorAll("[data-source-template]").forEach((button) => {
      button.addEventListener("click", () => {
        const template = document.getElementById(button.dataset.sourceTemplate || "");
        if (!template || !body || !title) return;

        returnFocus = button;
        title.textContent = button.dataset.sourceTitle || "Sources";
        body.replaceChildren(template.content.cloneNode(true));
        if (typeof modal.showModal === "function") {
          modal.showModal();
        } else {
          modal.setAttribute("open", "");
        }
        highlightCode();
        closeButton?.focus();
      });
    });

    closeButton?.addEventListener("click", () => {
      closeSourceModal();
    });

    modal.addEventListener("click", (event) => {
      if (event.target === modal) closeSourceModal();
    });

    modal.addEventListener("close", () => {
      body?.replaceChildren();
      returnFocus?.focus();
      returnFocus = null;
    });

    function closeSourceModal() {
      if (typeof modal.close === "function") {
        modal.close();
      } else {
        modal.removeAttribute("open");
        body?.replaceChildren();
        returnFocus?.focus();
        returnFocus = null;
      }
    }
  }
})();
