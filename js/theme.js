(() => {
  const root = document.documentElement;
  const button = document.querySelector("[data-theme-toggle]");

  if (!button) return;

  const preferredTheme = () => {
    if (root.dataset.theme) return root.dataset.theme;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  };

  const updateButton = (theme) => {
    const dark = theme === "dark";
    button.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
  };

  const setTheme = (theme, remember = true) => {
    root.dataset.theme = theme;
    updateButton(theme);

    if (remember) {
      try {
        localStorage.setItem("theme", theme);
      } catch (_) {
        // The theme still works when storage is unavailable.
      }
    }
  };

  setTheme(preferredTheme(), false);

  button.addEventListener("click", () => {
    setTheme(preferredTheme() === "dark" ? "light" : "dark");
  });
})();
