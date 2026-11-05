# The Toki Recur website

The public site is a static build. There is no server, no database and no
tracking: `dist/site/` is exactly what gets served, and it is rebuilt from
`site/` on every deploy.

## Layout

```
site/                  source
  index.html           landing page
  pricing.html         pricing page
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
  check-site-build.py  checks the built artifact; fails the deploy if wrong
  check-site-a11y.mjs  axe-core audit of the built site (and of candidates)
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
```

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

- clean URLs (`/pricing` → `pricing.html`, `/docs/` → `docs/index.html`);
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
- No third-party requests on page load: fonts are self-hosted and the video is
  a poster until somebody clicks it, and then it loads from YouTube's
  no-cookie host.
