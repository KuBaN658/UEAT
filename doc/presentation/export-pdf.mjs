import { createServer } from "node:http";
import { mkdir, readFile } from "node:fs/promises";
import { dirname, extname, join, normalize, resolve } from "node:path";
import process from "node:process";
import { chromium } from "playwright";
import { buildDiagramsFromMmd } from "./build-diagrams.mjs";

const HOST = "127.0.0.1";
const PORT = 4173;
const OUT_FILE = resolve(process.env.PDF_OUT || "diplomawork-presentation.pdf");
const ROOT = process.cwd();

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
};

function pathnameOnly(urlPath) {
  const s = urlPath || "/";
  return s.split("?")[0].split("#")[0] || "/";
}

const server = createServer(async (req, res) => {
  try {
    let pathname = pathnameOnly(req.url);
    if (pathname === "/") pathname = "/index.html";
    const path = normalize(decodeURIComponent(pathname)).replace(/^(\.\.[/\\])+/, "");
    const filePath = join(ROOT, path);
    const content = await readFile(filePath);
    const mime = MIME_TYPES[extname(filePath)] || "application/octet-stream";
    res.writeHead(200, { "Content-Type": mime });
    res.end(content);
  } catch {
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Not found");
  }
});

async function run() {
  buildDiagramsFromMmd();
  await mkdir(dirname(OUT_FILE), { recursive: true });
  await new Promise((resolveListen) => server.listen(PORT, HOST, resolveListen));

  const browser = await chromium.launch({
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-dev-shm-usage",
    ],
  });
  const page = await browser.newPage({
    viewport: { width: 1600, height: 900 },
  });
  page.setDefaultTimeout(60_000);

  await page.goto(`http://${HOST}:${PORT}/index.html?print-pdf`, {
    waitUntil: "load",
  });
  await page.waitForFunction(
    () => window.__presentationReady === true,
    { timeout: 60_000 },
  );
  await new Promise((r) => setTimeout(r, 500));

  await page.pdf({
    path: OUT_FILE,
    width: "1280px",
    height: "720px",
    printBackground: true,
    margin: { top: "0", right: "0", bottom: "0", left: "0" },
  });

  await browser.close();
  server.close();
  process.stdout.write(`Saved ${OUT_FILE}\n`);
}

run().catch((err) => {
  process.stderr.write(`PDF export failed: ${String(err)}\n`);
  server.close();
  process.exit(1);
});
