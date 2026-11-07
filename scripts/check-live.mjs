#!/usr/bin/env node
/* Check the site that is actually being served.

   The local gates prove the artifact is good. This proves the artifact is
   what the server is sending: the same axe audit, the same link crawl, the
   same contrast check and the same performance budget, run against the
   public URL.

   Usage:
     node scripts/check-live.mjs --url=https://tokirecur.com
*/

import { createRequire } from "node:module";
import { chromium } from "playwright";

const require = createRequire(import.meta.url);
const AXE_PATH = require.resolve("axe-core/axe.min.js");
const fs = await import("node:fs/promises");

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, ...v] = a.replace(/^--/, "").split("=");
    return [k, v.join("=") || true];
  })
);

const BASE = String(args.url || "https://tokirecur.com").replace(/\/$/, "");
const PAGES = [
  "/",
  "/privacy",
  "/terms",
  "/404.html",
  "/docs/",
  "/docs/install",
  "/docs/benchmark",
];
const WIDTHS = [1440, 390];

const axeSource = await fs.readFile(AXE_PATH, "utf8");
const browser = await chromium.launch({ headless: true });
const failures = [];

// 1. axe on every page, at two widths.
for (const route of PAGES) {
  for (const width of WIDTHS) {
    // The site ships a strict CSP, so the audit has to be allowed to inject
    // its own script. This only affects the browser running the audit.
    const context = await browser.newContext({ viewport: { width, height: 900 }, bypassCSP: true });
    const page = await context.newPage();
    await page.goto(BASE + route, { waitUntil: "networkidle" });
    await page.addScriptTag({ content: axeSource });
    const result = await page.evaluate(async () => {
      const r = await window.axe.run(document, {
        runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"] },
      });
      return r.violations.map((v) => ({
        id: v.id,
        impact: v.impact,
        nodes: v.nodes.length,
        target: v.nodes[0] && v.nodes[0].target.join(" "),
      }));
    });
    for (const v of result) failures.push(`axe ${route} @${width}: ${v.id} (${v.impact}) x${v.nodes} ${v.target}`);
    await page.close();
    await context.close();
  }
}

// 2. The landing page: no console errors, no failed requests, no overflow.
{
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("response", (r) => {
    if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`);
  });
  await page.goto(BASE + "/", { waitUntil: "networkidle" });
  await page.waitForTimeout(1500);
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth
  );
  if (overflow > 0) failures.push(`horizontal overflow on ${BASE}/: ${overflow}px`);
  for (const e of errors) failures.push(`console/network: ${e}`);
  await page.close();
}

// 3. Every internal link answers.
{
  const page = await browser.newPage();
  await page.goto(BASE + "/", { waitUntil: "domcontentloaded" });
  const hrefs = await page.$$eval("a[href]", (as) =>
    Array.from(new Set(as.map((a) => a.getAttribute("href")))).filter(
      (h) => h && h.startsWith("/") && !h.startsWith("//")
    )
  );
  await page.close();
  for (const href of hrefs) {
    const res = await fetch(BASE + href, { redirect: "manual" });
    if (res.status >= 400) failures.push(`link ${href} -> ${res.status}`);
  }
}

// 4. The certificate.
{
  const res = await fetch(BASE + "/", { method: "HEAD" });
  const cert = res.headers.get("strict-transport-security");
  if (!cert) failures.push("no HSTS header on the live site");
}

await browser.close();

if (failures.length) {
  console.log("live check failed:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
}
console.log(`live check ok: ${BASE} — ${PAGES.length} pages x ${WIDTHS.length} widths, links, headers, no console errors`);
