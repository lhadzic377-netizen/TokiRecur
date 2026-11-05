# Twenty directions, and the one that shipped

The site was redrawn twenty times before it was rebuilt. Every candidate
below is a real page — the production `index.html` with a different visual
system layered over it — so the twenty could be rendered, measured and
compared on equal terms: same words, same claims, same accessibility, one
variable changed.

```bash
python3 scripts/design-explorer.py     # build dist/designs/<slug>/
node scripts/shoot-designs.mjs         # render them to /tmp/opencode/designs
```

`scripts/shoot-designs.mjs` writes a screenshot per candidate, a contact
sheet, and a table of the numbers a screenshot cannot show: overflow, the
headline's typeface and size, and the height of the page.

## The candidates

| # | Slug | The idea |
|---|------|----------|
| 01 | `terminal-native` | The page is a TUI: mono, amber phosphor, window chrome, `[ brackets ]` for actions. |
| 02 | `instrument-editorial` | A magazine about a system: Instrument Serif, oxblood, rules instead of boxes. |
| 03 | `swiss-graphite` | Twelve columns, one red, no decoration. |
| 04 | `dashboard-console` | The landing page as an instrument panel. |
| 05 | `aurora-depth` | A rendered field of light behind frosted glass. |
| 06 | `blueprint-cyan` | Technical drawing: graph paper, dimension lines, plotter cyan. |
| 07 | `bauhaus-primary` | Primary colours used as structure, not decoration. |
| 08 | `paper-light` | The site in daylight: warm paper, serif text, green ink. |
| 09 | `brutal-raw` | No radius, no shadow, no politeness. |
| 10 | `cinema-dark` | One enormous image, one enormous line of type. |
| 11 | `dataviz-hero` | The hero is the measurement: what was skipped, drawn. |
| 12 | `split-console` | Statement on the left, live terminal on the right. |
| 13 | `swiss-minimal` | One column, enormous quiet, a single accent. |
| 14 | `crt-phosphor` | Amber phosphor, scanlines, a screen you can lean into. |
| 15 | `docs-first` | The landing page as an index into the documentation. |
| 16 | `bento-modular` | Tiles of different sizes, each with its own colour. |
| 17 | `newspaper` | Columns, rules, a masthead and a dateline. |
| 18 | `neon-arcade` | Saturated light on a purple-black field. |
| 19 | `riso-print` | Two spot inks on warm stock, misregistered on purpose. |
| 20 | `ink-quiet` | Black, Geist, one yellow — the current direction, finished. |

## Why 20 won

The contact sheet decides it. The other nineteen are beautiful in the
first screen and then change their mind halfway down the page: the
editorial hero sits above a page of default dark cards, the cinematic
hero sits above a page of default dark cards, the bento and the aurora
and the neon all do the same. They are a *hero* with a default page
attached.

`ink-quiet` was the only candidate whose whole page held one voice —
hero, demo, figures, features, benchmark table, footer — and the only one
that matches the audience. Toki Recur is infrastructure. It should look
like infrastructure: dark, precise, calm, with the terminal in front and
the numbers behind it.

Three things from the other candidates were worth keeping:

- **`dataviz-hero`'s chart** — the idea that a claim can be *drawn* rather
  than asserted. It became the prefix matcher: a band that animates the
  actual mechanism, the prefix walking the prompt, the matched tokens
  turning into cache hits and the tail turning into work.
- **`terminal-native`'s window chrome** and honest monospace, which the
  page already had.
- **`swiss-graphite`'s discipline** — the figures line up like a table
  instead of floating in four cards.

## What the winner added

- The **prefix matcher** (`site/mechanism.js`): the product's mechanism,
  animated, with the measured hit rate and nothing invented.
- The **backend strip**: a slow list of the backends this actually runs
  on — CUDA, ROCm, SYCL, OpenVINO, Vulkan, Metal, CPU.
- A **ledger** for the figures, a **printed table** for the benchmarks, and
  a full-bleed band under the demo so the page has a pulse.
- **Small motions** (`site/motion.js`): the cards light up where the
  pointer is, the hero drifts as you leave it, the reuse diagram carries
  something along its path. Transform- and opacity-only, and all of them
  stand down under `prefers-reduced-motion`.

None of it is decoration for its own sake. Every band on the page says
something the page needs to say.

## Fonts

The candidates needed families that the production site does not use.
They live in `site/designs/fonts/` with the `@font-face` block in
`site/designs/fonts-designs.css`, and they are **not** part of the
deployed site: `scripts/build-site.py` copies `site/fonts/` (Geist and
Geist Mono only) and never `site/designs/`.

The shipped site uses Geist and Geist Mono, self-hosted, latin subset,
with the OFL licence in `site/fonts/OFL.txt`.
