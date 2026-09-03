// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

(() => {
  const storageKey = "simjit-explorer-theme";

  function normalizeTheme(theme) {
    return theme === "dark" || theme === "light" ? theme : "light";
  }

  function readStoredTheme() {
    try {
      return normalizeTheme(localStorage.getItem(storageKey));
    } catch (_) {
      return "light";
    }
  }

  function applyTheme(theme) {
    const normalized = normalizeTheme(theme);
    document.documentElement.dataset.theme = normalized;
    return normalized;
  }

  function storeTheme(theme) {
    const normalized = applyTheme(theme);
    try {
      localStorage.setItem(storageKey, normalized);
    } catch (_) {
      // Browser storage can be disabled in private or locked-down contexts.
    }
    return normalized;
  }

  function syncThemeButton(button) {
    if (!button) return;
    const theme = normalizeTheme(document.documentElement.dataset.theme);
    button.textContent = theme === "dark" ? "Light theme" : "Dark theme";
    button.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
    button.title = theme === "dark" ? "Switch to light theme" : "Switch to dark theme";
  }

  function syncAllThemeButtons() {
    document.querySelectorAll("[data-theme-toggle]").forEach(syncThemeButton);
  }

  function bindThemeToggle(button) {
    applyTheme(readStoredTheme());
    syncThemeButton(button);
    if (!button) return;
    button.addEventListener("click", () => {
      const next = normalizeTheme(document.documentElement.dataset.theme) === "dark" ? "light" : "dark";
      storeTheme(next);
      syncAllThemeButtons();
    });
  }

  window.simjitTheme = {
    applyTheme,
    bindThemeToggle,
    readStoredTheme,
    storageKey,
    syncAllThemeButtons,
    syncThemeButton,
  };

  try {
    applyTheme(readStoredTheme());
  } catch (_) {
    document.documentElement.dataset.theme = "light";
  }

  window.addEventListener("storage", (event) => {
    if (event.key !== storageKey) return;
    applyTheme(event.newValue);
    syncAllThemeButtons();
  });
})();
