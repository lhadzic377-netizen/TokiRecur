#!/usr/bin/env node
/* Report the contrast of the text on the page.

   Dark pages go wrong in the small print: the muted grey that reads fine
   on the hero reads as mud in a table cell, and a mono label at nine
   pixels is unreadable.  This walks the rendered page, resolves the
   effective background behind each text node, and prints the worst
   offenders.

   Usage:
     node scripts/check-contrast.mjs [--url=http://127.0.0.1:8903/]
*/

import { chromium } from "playwright";

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, ...v] = a.replace(/^--/, "").split("=");
    return [k, v.join("=") || true];
  })
);

const url = String(args.url || "http://127.0.0.1:8903/");

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
await page.goto(url, { waitUntil: "networkidle" });
await page.waitForTimeout(600);

const rows = await page.evaluate(() => {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 1;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  // Chrome keeps oklch() as it was written, so the colour has to go through
  // a canvas to become the sRGB bytes the ratio is computed from.
  const toRGBA = (color) => {
    ctx.clearRect(0, 0, 1, 1);
    ctx.fillStyle = "#000";
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, 1, 1);
    const d = ctx.getImageData(0, 0, 1, 1).data;
    return [d[0], d[1], d[2], d[3] / 255];
  };

  const lum = (rgb) => {
    const f = rgb.map((v) => {
      const s = v / 255;
      return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * f[0] + 0.7152 * f[1] + 0.0722 * f[2];
  };
  const ratio = (a, b) => {
    const la = lum(a);
    const lb = lum(b);
    const [hi, lo] = la > lb ? [la, lb] : [lb, la];
    return (hi + 0.05) / (lo + 0.05);
  };
  const blend = (fg, bg, alpha) => fg.map((v, i) => v * alpha + bg[i] * (1 - alpha));

  const effectiveBg = (el) => {
    let node = el;
    const stack = [];
    while (node && node !== document.documentElement) {
      const cs = getComputedStyle(node);
      const rgba = toRGBA(cs.backgroundColor);
      if (rgba[3] > 0) {
        stack.push({ bg: rgba.slice(0, 3), alpha: rgba[3] });
        if (rgba[3] >= 0.999) break;
      }
      node = node.parentElement;
    }
    let out = [7, 9, 17];
    for (const layer of stack.reverse()) out = blend(layer.bg, out, layer.alpha);
    return out;
  };

  const out = [];
  document.querySelectorAll("body *").forEach((el) => {
    const text = Array.from(el.childNodes)
      .filter((n) => n.nodeType === 3)
      .map((n) => n.textContent.trim())
      .join(" ")
      .trim();
    if (!text) return;
    const cs = getComputedStyle(el);
    if (cs.visibility === "hidden" || cs.display === "none" || Number(cs.opacity) < 0.1) return;
    const rect = el.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const fg = toRGBA(cs.color);
    if (fg[3] < 0.1) return;
    const bg = effectiveBg(el);
    const size = parseFloat(cs.fontSize);
    const bold = Number(cs.fontWeight) >= 700;
    const large = size >= 24 || (size >= 18.66 && bold);
    const need = large ? 3 : 4.5;
    const r = ratio(blend(fg.slice(0, 3), bg, fg[3]), bg);
    if (r < need) {
      out.push({
        text: text.slice(0, 44),
        cls: (el.className && el.className.toString ? el.className.toString() : el.tagName).slice(0, 34),
        size: Math.round(size),
        ratio: Number(r.toFixed(2)),
        need,
      });
    }
  });
  return out.sort((a, b) => a.ratio - b.ratio);
});

const seen = new Set();
for (const r of rows) {
  const key = `${r.cls}|${r.text}`;
  if (seen.has(key)) continue;
  seen.add(key);
  console.log(
    `${String(r.ratio).padStart(5)} (need ${r.need})  ${String(r.size).padStart(2)}px  ${r.cls.padEnd(34)}  ${r.text}`
  );
}
console.log(rows.length ? `\n${rows.length} low-contrast text nodes` : "no low-contrast text found");

await browser.close();
