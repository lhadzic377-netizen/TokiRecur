# The Toki Recur website

The public site is a static build. There is no server, no database and no
tracking: `dist/site/` is exactly what gets served, and it is rebuilt from
`site/` on every deploy.

## Layout

```
site/                  source
  index.html           landing page
  404.html             not-found page (served with a 404 status)
  style.css            design system + landing page
  fonts.css            self-hosted Geist / Geist Mono
  head.js              runs before paint: adds .js-ready, sets the no-JS fallback
  app.js               nav, tabs, copy buttons, scroll reveals, count-ups
  mechanism.js         the prefix matcher: the product's mechanism, animated
  motion.js            the small motions: pointer light, hero drift
  tui.js               the launcher simulation (keyboard driven)
  docs/                documentation pages
  img/                 screenshots and marks
  fonts/               Geist woff2 subsets + the OFL license
  designs/             twenty visual directions (never deployed)
scripts/
  build-site.py        builds dist/site/ (the deployable artifact)
  serve-site.py        serves dist/site with the server's URL rules
  check-site-build.py  checks the built artifact; fails the deploy if wrong
  check-site-a11y.mjs  axe-core audit of the built site (and of candidates)
  check-site-perf.mjs  wire weight, LCP and CLS against the budget
  check-contrast.mjs   reports the lowest-contrast text on the page
  check-links.mjs      crawls the built site for links that 404
  prune-css.py         removes the CSS for sections the redesign replaced
  design-explorer.py   assembles the twenty design candidates
  shoot-designs.mjs    renders the candidates so they can be compared
  install-site.sh      installs Caddy, deploys the artifact, wires the chat
  Caddyfile.template   the server config the installer fills in
  build-site-zip.sh    packages the artifact + installer as dist/toki-site.zip
```

## Build

```bash
python3 scripts/build-site.py            # -> dist/site/
python3 scripts/check-site-build.py      # -> refuses a broken artifact
python3 scripts/serve-site.py            # -> http://127.0.0.1:8903/
```

Look at the site with `serve-site.py`, not with a generic static server.
The deployment resolves a URL the way Caddy does: the path, then the path
with `.html`, then the path as a directory index. That is what makes
`/privacy`, `/terms` and `/docs/install` work. A server that does not do
that returns 404 for pages that are in the build.

The build:

- copies only the files that may be public (no `server.py`, no `data/`);
- fingerprints every asset and rewrites the references (`/style.css?v=…`);
- rewrites the canonical URL, `og:url` and the share image for the domain
  being deployed to (`--site-url`, default `https://tokirecur.com`);
- fails when an internal link or asset reference does not resolve;
- writes `sitemap.xml` and `robots.txt`.

## Deploy

```bash
sudo ./scripts/install-site.sh --domain tokirecur.com --email you@example.com
```

The installer builds the site, deploys it as an immutable release under
`/opt/toki-site/releases/<timestamp>`, and points `/opt/toki-site/current` at
it, so a rollback is one `ln -sfn` away. It then writes `/etc/caddy/Caddyfile`
from the template and validates it before Caddy is restarted.

Prove the whole path without touching the machine:

```bash
./scripts/install-site.sh --dry-run --domain tokirecur.com
```

The dry run builds the site, renders the Caddyfile and writes the systemd
unit into a temporary directory, then prints what a real run would do.

## What the server does

`scripts/Caddyfile.template` gives the site:

- clean URLs (`/privacy` → `privacy.html`, `/docs/install` → `docs/install.html`),
  which are the only addresses the site links to, the sitemap names and the
  canonical tags declare;
- a real 404 (`/404.html`, served with a 404 status);
- `no-cache` for HTML, `immutable` for fingerprinted assets;
- `Content-Security-Policy`, `Strict-Transport-Security`, `X-Content-Type-Options`,
  `Referrer-Policy` and `Permissions-Policy` on every response;
- a `www` → apex redirect and a `chat.<domain>` reverse proxy for the web chat.

## Design

The page was redrawn twenty times before it shipped. The candidates, the
screenshots and the reasoning are in `site/designs/README.md`; the winning
direction is `ink-quiet`, and it is what `style.css` implements.

```bash
python3 scripts/design-explorer.py     # assemble dist/designs/<slug>/
node scripts/shoot-designs.mjs         # render and measure them
```

`site/designs/` is source, not output: the build never copies it, so the
candidate stylesheets and their typefaces never reach the server.

## Deliberately not deployed

- No wishlist counter, no API, no visitor state: the site collects nothing.
- No analytics of any kind.
- No third-party requests on page load: the fonts and the images are served
  from the same machine as the page, and there is no video to embed.
