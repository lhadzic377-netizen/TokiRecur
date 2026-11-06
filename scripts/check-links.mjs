#!/usr/bin/env node
/* Crawl the built site and report every internal link that does not answer.

   The server that ships the site resolves a URL the way Caddy does:
   /path, then /path.html, then /path/index.html.  A link that only works
   on the local Python server (or only in a browser's address bar) is a
   404 in production, so the crawl uses the production rule.

   Usage:
     node scripts/check-links.mjs --root=dist/site
*/

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize } from "node:path";

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, ...v] = a.replace(/^--/, "").split("=");
    return [k, v.join("=") || true];
  })
);

const ROOT = join(process.cwd(), String(args.root || "dist/site"));

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
  ".xml": "application/xml; charset=utf-8",
};

async function exists(p) {
  try {
    return (await stat(p)).isFile();
  } catch {
    return false;
  }
}

// Caddy: try_files {path} {path}.html {path}/index.html
async function resolve(pathname) {
  const clean = normalize(decodeURIComponent(pathname)).replace(/^(\.\.[/\\])+/, "");
  const base = join(ROOT, clean);
  const candidates = [base];
  if (!extname(base)) {
    candidates.push(`${base}.html`, join(base, "index.html"));
  }
  for (const c of candidates) {
    if (await exists(c)) return c;
  }
  return null;
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  const file = await resolve(url.pathname);
  if (!file) {
    const notFound = join(ROOT, "404.html");
    if (await exists(notFound)) {
      const body = await readFile(notFound);
      res.writeHead(404, { "content-type": TYPES[".html"] });
      res.end(body);
      return;
    }
    res.writeHead(404);
    res.end("not found");
    return;
  }
  const body = await readFile(file);
  res.writeHead(200, { "content-type": TYPES[extname(file)] || "application/octet-stream" });
  res.end(body);
});

await new Promise((r) => server.listen(0, "127.0.0.1", r));
const origin = `http://127.0.0.1:${server.address().port}`;

// Every page in the tree.
const pages = [];
async function walk(dir) {
  for (const entry of await (await import("node:fs/promises")).readdir(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) await walk(full);
    else if (entry.name.endsWith(".html")) pages.push(full);
  }
}
await walk(ROOT);

const links = new Map(); // url -> [pages that link to it]
const assets = new Map();

for (const page of pages) {
  const html = await readFile(page, "utf8");
  const from = page.replace(`${ROOT}/`, "");
  for (const m of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
    const raw = m[1];
    if (/^(https?:|mailto:|tel:|data:|#|\/\/)/.test(raw)) continue;
    const [path] = raw.split("#");
    if (!path) continue;
    if (path.startsWith("/")) {
      const list = links.get(path) || [];
      list.push(from);
      links.set(path, list);
    } else {
      const abs = `/${path.replace(/^\.\//, "")}`;
      const list = assets.get(abs) || [];
      list.push(from);
      assets.set(abs, list);
    }
  }
}

const broken = [];
const checked = new Set();

for (const [url, from] of [...links, ...assets]) {
  if (checked.has(url)) continue;
  checked.add(url);
  const res = await fetch(origin + url, { redirect: "manual" });
  if (res.status !== 200) broken.push({ url, status: res.status, from: [...new Set(from)] });
}

// The sitemap is a promise about what exists; check it too.
const sitemap = await readFile(join(ROOT, "sitemap.xml"), "utf8");
const sitemapUrls = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => m[1]);
const sitemapBroken = [];
for (const raw of sitemapUrls) {
  const path = new URL(raw).pathname;
  const res = await fetch(origin + path, { redirect: "manual" });
  if (res.status !== 200) sitemapBroken.push({ url: path, status: res.status });
}

console.log(`crawled ${pages.length} pages, ${checked.size} internal urls`);
if (broken.length) {
  console.log("\nbroken internal links:");
  for (const b of broken) console.log(`  ${b.status}  ${b.url}   (from ${b.from.join(", ")})`);
} else {
  console.log("no broken internal links");
}
if (sitemapBroken.length) {
  console.log("\nbroken sitemap urls:");
  for (const b of sitemapBroken) console.log(`  ${b.status}  ${b.url}`);
} else {
  console.log("sitemap: every url answers 200");
}

server.close();
process.exit(broken.length || sitemapBroken.length ? 1 : 0);
