#!/usr/bin/env node
/**
 * Render the design candidates so they can be compared side by side.
 *
 *   node scripts/shoot-designs.mjs                 # every candidate
 *   node scripts/shoot-designs.mjs terminal-native # one candidate
 *
 * Writes PNGs to /tmp/opencode/designs/<slug>-<viewport>.png and a contact
 * sheet per candidate. Also reports the objective numbers a screenshot cannot
 * show: overflow, contrast of the headline pair, and how long the page is.
 */

import { createServer } from "node:http";
import { readFile, stat, mkdir } from "node:fs/promises";
import { join, extname, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { dirname } from "node:path";
import { chromium } from "playwright";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const DESIGNS = join(ROOT, "dist", "designs");
const OUT = "/tmp/opencode/designs";

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".woff2": "font/woff2",
  ".txt": "text/plain; charset=utf-8",
  ".xml": "application/xml",
};

function serve(root) {
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, "http://localhost");
    let path = normalize(decodeURIComponent(url.pathname)).replace(/^(\.\.[/\\])+/, "");
    let file = join(root, path);
    try {
      const info = await stat(file);
      if (info.isDirectory()) file = join(file, "index.html");
    } catch {
      file = join(root, "404.html");
    }
    try {
      const body = await readFile(file);
      res.writeHead(200, { "content-type": TYPES[extname(file)] ?? "application/octet-stream" });
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

const only = process.argv.slice(2).filter((a) => !a.startsWith("-"));

async function main() {
  await mkdir(OUT, { recursive: true });
  const entries = await import("node:fs/promises").then((fs) => fs.readdir(DESIGNS, { withFileTypes: true }));
  const dirs = entries.filter((e) => e.isDirectory()).map((e) => e.name);
  const targets = only.length ? only : dirs;

  const browser = await chromium.launch({ headless: true });
  const rows = [];

  for (const slug of targets) {
    const root = join(DESIGNS, slug);
    const server = await serve(root);
    const { port } = server.address();
    const row = { slug };

    for (const [name, viewport] of [
      ["desktop", { width: 1440, height: 900 }],
      ["mobile", { width: 390, height: 844 }],
    ]) {
      const context = await browser.newContext({ viewport, deviceScaleFactor: 1 });
      const page = await context.newPage();
      const errors = [];
      page.on("console", (m) => {
        if (m.type() === "error") errors.push(m.text());
      });
      await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "networkidle" });
      await page.waitForTimeout(700);
      await page.evaluate(() => {
        document.querySelectorAll(".reveal").forEach((el) => el.classList.add("in-view"));
      });
      await page.waitForTimeout(450);

      if (name === "desktop") {
        row.metrics = await page.evaluate(() => {
          const h1 = document.querySelector("h1");
          const body = document.body;
          const cs = getComputedStyle(h1);
          const lede = document.querySelector(".lede");
          const lcs = getComputedStyle(lede);
          const link = document.querySelector(".links a");
          return {
            overflow: document.documentElement.scrollWidth > window.innerWidth + 1,
            scrollWidth: document.documentElement.scrollWidth,
            pageHeight: document.documentElement.scrollHeight,
            h1Family: cs.fontFamily.split(",")[0].replace(/"/g, ""),
            h1Size: cs.fontSize,
            h1Weight: cs.fontWeight,
            h1Color: cs.color,
            h1Bg: getComputedStyle(document.body).backgroundColor,
            ledeColor: lcs.color,
            ledeSize: lcs.fontSize,
            navSize: link ? getComputedStyle(link).fontSize : "-",
            sections: document.querySelectorAll("main > section").length,
          };
        });
      }

      await page.screenshot({ path: join(OUT, `${slug}-${name}.png`), fullPage: name === "desktop" });
      if (name === "desktop") await page.screenshot({ path: join(OUT, `${slug}-hero.png`) });
      if (errors.length) row[name + "Errors"] = errors.slice(0, 3);
      await context.close();
    }

    server.close();
    rows.push(row);
    console.log(`shot ${slug}`);
  }

  console.log("\n" + "slug".padEnd(22) + "overflow  h1 font".padEnd(24) + "h1 size   lede size  page height");
  for (const r of rows) {
    const m = r.metrics || {};
    console.log(
      r.slug.padEnd(22) +
        String(m.overflow).padEnd(9) +
        `${m.h1Family} ${m.h1Weight}`.padEnd(24) +
        `${m.h1Size}`.padEnd(10) +
        `${m.ledeSize}`.padEnd(11) +
        m.pageHeight,
    );
  }
  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
