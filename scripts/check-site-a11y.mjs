#!/usr/bin/env node
/**
 * Accessibility audit for the built site.
 *
 *   node scripts/check-site-a11y.mjs
 *
 * Serves dist/site the way Caddy does (clean URLs, a real 404) and runs
 * axe-core over every page at three widths. Exits non-zero on any
 * WCAG 2.2 A/AA violation, so it can sit in CI.
 *
 * Requires: npm install playwright axe-core && npx playwright install chromium
 */

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { dirname } from "node:path";
import { chromium } from "playwright";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const AXE_PATH = require.resolve("axe-core/axe.min.js");

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const DIST = join(ROOT, "dist", "site");

const PAGES = [
  "/",
  "/privacy.html",
  "/terms.html",
  "/404.html",
  "/docs/",
  "/docs/install.html",
  "/docs/quickstart.html",
  "/docs/commands.html",
  "/docs/interfaces.html",
  "/docs/architecture.html",
  "/docs/api.html",
  "/docs/benchmark.html",
  "/docs/deploy.html",
  "/docs/development.html",
];

const WIDTHS = [1440, 1024, 390];

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".webp": "image/webp",
  ".woff2": "font/woff2",
  ".txt": "text/plain; charset=utf-8",
  ".xml": "application/xml",
};

async function resolveFile(pathname, root = DIST) {
  const clean = normalize(decodeURIComponent(pathname)).replace(/^(\.\.[/\\])+/, "");
  const candidates = [
    join(root, clean),
    join(root, `${clean}.html`),
    join(root, clean, "index.html"),
  ];
  for (const candidate of candidates) {
    try {
      const info = await stat(candidate);
      if (info.isFile()) return candidate;
    } catch {
      /* try the next one */
    }
  }
  return null;
}

function startServer(root) {
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, "http://localhost");
    let file = await resolveFile(url.pathname, root);
    let status = 200;
    if (!file) {
      file = join(root, "404.html");
      status = 404;
    }
    try {
      const body = await readFile(file);
      res.writeHead(status, {
        "content-type": TYPES[extname(file)] ?? "application/octet-stream",
      });
      res.end(body);
    } catch {
      res.writeHead(404, { "content-type": "text/plain" });
      res.end("not found");
    }
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve(server));
  });
}

async function settle(page) {
  await page.evaluate(async () => {
    const animations = document.getAnimations ? document.getAnimations() : [];
    // The looping animations never finish, so waiting for all of them would
    // wait forever. Only the finite ones are awaited, and the whole thing is
    // capped.
    const finite = animations.filter((a) => {
      const timing = a.effect && a.effect.getComputedTiming ? a.effect.getComputedTiming() : null;
      return !timing || timing.iterations !== Infinity;
    });
    await Promise.race([
      Promise.all(finite.map((a) => a.finished.catch(() => {}))),
      new Promise((resolve) => setTimeout(resolve, 1500)),
    ]);
  });
  await page.waitForTimeout(250);
}

async function main() {
  const rootArg = process.argv.find((a) => a.startsWith("--root="));
  const root = rootArg ? rootArg.split("=")[1] : DIST;
  const server = await startServer(root);
  const { port } = server.address();
  const browser = await chromium.launch({ headless: true });
  let failures = 0;

  for (const width of WIDTHS) {
    const context = await browser.newContext({ viewport: { width, height: 900 } });
    for (const route of PAGES) {
      const page = await context.newPage();
      await page.goto(`http://127.0.0.1:${port}${route}`, { waitUntil: "networkidle" });
      await settle(page);
      await page.addScriptTag({ path: AXE_PATH });
      const result = await page.evaluate(async () =>
        axe.run(document, {
          runOnly: {
            type: "tag",
            values: ["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"],
          },
        }),
      );
      if (result.violations.length) {
        failures += result.violations.length;
        console.error(`\n${width}px ${route}`);
        for (const violation of result.violations) {
          console.error(`  [${violation.impact}] ${violation.id}: ${violation.help}`);
          for (const node of violation.nodes.slice(0, 5)) {
            console.error(`      ${node.target.join(" ")}`);
            console.error(`      ${node.failureSummary.split("\n")[0]}`);
          }
        }
      }
      await page.close();
    }
    await context.close();
  }

  await browser.close();
  server.close();

  if (failures) {
    console.error(`\n${failures} accessibility violations`);
    process.exit(1);
  }
  console.log(`accessibility check ok: ${PAGES.length} pages x ${WIDTHS.length} widths, no WCAG 2.2 A/AA violations`);
  process.exit(0);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
