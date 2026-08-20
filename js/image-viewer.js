(() => {
  const dialog = document.querySelector("#image-viewer");
  const preview = dialog?.querySelector("[data-image-preview]");
  const caption = dialog?.querySelector("[data-image-caption]");

  if (!dialog || !preview || !caption || typeof dialog.showModal !== "function") return;

  document.addEventListener("click", (event) => {
    const image = event.target.closest("main img:not([data-no-zoom])");
    if (!image || image.closest("[data-no-zoom]")) return;

    event.preventDefault();
    preview.src = image.currentSrc || image.src;
    preview.alt = image.alt || "";
    caption.textContent = image.dataset.imageCaption || image.alt || "";
    dialog.showModal();
  });

  dialog.addEventListener("click", () => {
    dialog.close();
  });

  dialog.addEventListener("close", () => {
    preview.removeAttribute("src");
    caption.textContent = "";
  });
})();
