/**
 * Loads .mmd files, renders Mermaid on each pre.mermaid only, then starts Reveal.
 * PDF export waits for window.__presentationReady === true.
 */
(async function () {
  async function loadDiagramSources() {
    const nodes = document.querySelectorAll("pre.mermaid[data-mmd]");
    await Promise.all(
      Array.from(nodes).map(async (pre) => {
        const url = pre.getAttribute("data-mmd");
        const res = await fetch(url, { cache: "no-store" });
        if (!res.ok) {
          throw new Error("Failed to load diagram: " + url + " (" + res.status + ")");
        }
        pre.textContent = await res.text();
      }),
    );
  }

  mermaid.initialize({
    startOnLoad: false,
    theme: "neutral",
    securityLevel: "loose",
    fontFamily: "system-ui, Segoe UI, sans-serif",
  });

  await loadDiagramSources();

  const diagramNodes = document.querySelectorAll("pre.mermaid");
  await mermaid.run({ nodes: diagramNodes });

  if (document.fonts && document.fonts.ready) {
    await document.fonts.ready;
  }

  await Reveal.initialize({
    hash: true,
    slideNumber: "c/t",
    controls: true,
    progress: true,
    center: false,
    transition: "slide",
    width: 1280,
    height: 720,
    margin: 0.04,
    plugins: [RevealNotes],
    pdfSeparateFragments: false,
  });

  window.__presentationReady = true;
})().catch(function (err) {
  console.error(err);
  window.__presentationError = String(err);
  window.__presentationReady = true;
});
