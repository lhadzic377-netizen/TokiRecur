#!/usr/bin/env node
/* Measure the three numbers the site promises itself.

   Wire weight   every byte the browser pulls for the landing page, including
                 the fonts and the images that lazy-load as you scroll.
   LCP           the largest contentful paint, on a throttled connection.
   CLS           the cumulative layout shift while the page settles.

   Usage:
     node scripts/check-site-perf.mjs --root=dist/site [--url=http://127.0.0.1:8903/]

   The server must already be serving the built tree.  The script exits
   non-zero when a budget is blown, so it can be used as a gate.
*/

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { chromium } from "playwright";

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, ...v] = a.replace(/^--/, "").split("=");
    return [k, v.join("=") || true];
  })
);

const ROOT = join(process.cwd(), String(args.root || "dist/site"));

const BUDGET = {
  wireKiB: 900,
  lcpMs: 1200,
  cls: 0.01,
};

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".woff2": "font/woff2",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".txt": "text/plain; charset=utf-8",
};

const server = createServer(async (req, res) => {
  try {
    let path = decodeURIComponent(new URL(req.url, "http://x").pathname);
    if (path.endsWith("/")) path += "index.html";
    const file = join(ROOT, normalize(path).replace(/^(\.\.[/\\])+/, ""));
    const info = await stat(file);
    if (!info.isFile()) throw new Error("not a file");
    const body = await readFile(file);
    res.writeHead(200, {
      "content-type": TYPES[extname(file)] || "application/octet-stream",
      "content-length": body.length,
      "cache-control": "public, max-age=31536000, immutable",
    });
    res.end(body);
  } catch {
    res.writeHead(404, { "content-type": "text/plain" });
    res.end("not found");
  }
});

await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;
const url = String(args.url || `http://127.0.0.1:${port}/`);

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 1,
});

const page = await context.newPage();

const bytes = new Map();
let wire = 0;

page.on("response", async (response) => {
  const request = response.request();
  if (request.resourceType() === "document") return;
  const size = Number(response.headers()["content-length"] || 0);
  const body = await response.body().catch(() => null);
  const n = body ? body.length : size;
  bytes.set(response.url(), n);
  wire += n;
});

// A throttled profile: the number a reader on a laptop tether feels.
const client = await context.newCDPSession(page);
await client.send("Network.enable");
await client.send("Network.emulateNetworkConditions", {
  offline: false,
  latency: 40,
  downloadThroughput: (10 * 1024 * 1024) / 8,
  uploadThroughput: (2 * 1024 * 1024) / 8,
});
await client.send("Emulation.setCPUThrottlingRate", { rate: 2 });

await page.addInitScript(() => {
  window.__cls = 0;
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      if (!entry.hadRecentInput) window.__cls += entry.value;
    }
  }).observe({ type: "layout-shift", buffered: true });
});

await page.goto(url, { waitUntil: "load" });

const lcp = await page.evaluate(
  () =>
    new Promise((resolve) => {
      let value = 0;
      new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) value = entry.startTime;
      }).observe({ type: "largest-contentful-paint", buffered: true });
      setTimeout(() => resolve(value), 600);
    })
);

// Walk the page so the lazy images are pulled too, then settle.
const height = await page.evaluate(() => document.documentElement.scrollHeight);
for (let y = 0; y < height; y += 700) {
  await page.evaluate((v) => window.scrollTo(0, v), y);
  await page.waitForTimeout(60);
}

// Every image the page can show, loaded: the weight a reader pays after
// reading to the end.
await page.evaluate(() => {
  document.querySelectorAll("img").forEach((img) => {
    img.loading = "eager";
  });
});
await page.waitForFunction(
  () => Array.from(document.images).every((img) => img.complete && img.naturalWidth > 0),
  null,
  { timeout: 30000 }
);
await page.waitForTimeout(800);

const cls = await page.evaluate(() => window.__cls || 0);
const paint = await page.evaluate(() => {
  const lcpEntry = performance.getEntriesByType("largest-contentful-paint");
  const nav = performance.getEntriesByType("navigation")[0];
  return {
    lcp: lcpEntry.length ? lcpEntry[lcpEntry.length - 1].startTime : 0,
    domContentLoaded: nav ? nav.domContentLoadedEventEnd : 0,
    load: nav ? nav.loadEventEnd : 0,
  };
});

const total = wire / 1024;
const result = {
  wireKiB: Math.round(total),
  lcpMs: Math.round(Math.max(lcp, paint.lcp)),
  cls: Number(cls.toFixed(4)),
  requests: bytes.size,
  largest: [...bytes.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([u, n]) => [u.replace(/^https?:\/\/[^/]+/, ""), Math.round(n / 1024) + " KiB"]),
};

const fails = [];
if (result.wireKiB > BUDGET.wireKiB) fails.push(`wire ${result.wireKiB} KiB > ${BUDGET.wireKiB}`);
if (result.lcpMs > BUDGET.lcpMs) fails.push(`LCP ${result.lcpMs} ms > ${BUDGET.lcpMs}`);
if (result.cls > BUDGET.cls) fails.push(`CLS ${result.cls} > ${BUDGET.cls}`);

console.log(JSON.stringify(result, null, 2));
console.log(
  fails.length
    ? `performance budget exceeded: ${fails.join("; ")}`
    : "performance budget ok"
);

await browser.close();
server.close();
process.exit(fails.length ? 1 : 0);
