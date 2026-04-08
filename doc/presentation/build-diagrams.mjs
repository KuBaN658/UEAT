import { existsSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import process from "node:process";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const ROOT = dirname(fileURLToPath(import.meta.url));

export function buildDiagramsFromMmd() {
  const mmdcCli = join(
    ROOT,
    "node_modules",
    "@mermaid-js",
    "mermaid-cli",
    "src",
    "cli.js",
  );
  if (!existsSync(mmdcCli)) {
    throw new Error(
      "Mermaid CLI not found at " + mmdcCli + "; run npm install in doc/presentation",
    );
  }
  const diagramsDir = join(ROOT, "diagrams");
  const files = readdirSync(diagramsDir).filter((f) => f.endsWith(".mmd"));
  for (const f of files) {
    const inPath = join(diagramsDir, f);
    const outPath = join(diagramsDir, f.replace(/\.mmd$/i, ".svg"));
    execFileSync(
      process.execPath,
      [mmdcCli, "-i", inPath, "-o", outPath, "-b", "transparent", "--quiet"],
      { cwd: ROOT, stdio: "inherit", env: process.env },
    );
  }
}

const thisFile = fileURLToPath(import.meta.url);
if (process.argv[1] && resolve(process.argv[1]) === resolve(thisFile)) {
  buildDiagramsFromMmd();
}
