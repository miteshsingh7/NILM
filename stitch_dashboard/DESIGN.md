---
name: Telemetry Rack & Oscilloscope Console
colors:
  surface: '#1e1000'
  surface-dim: '#1e1000'
  surface-bright: '#513300'
  surface-container-lowest: '#170c00'
  surface-container-low: '#291800'
  surface-container: '#2f1c00'
  surface-container-high: '#3c2500'
  surface-container-highest: '#4b2f00'
  on-surface: '#ffddb4'
  on-surface-variant: '#e5bcc4'
  inverse-surface: '#ffddb4'
  inverse-on-surface: '#452b00'
  outline: '#ac878f'
  outline-variant: '#5c3f45'
  surface-tint: '#ffb1c3'
  primary: '#ffb1c3'
  on-primary: '#66002c'
  primary-container: '#ff4b89'
  on-primary-container: '#590026'
  inverse-primary: '#bb0058'
  secondary: '#d3fbff'
  on-secondary: '#00363a'
  secondary-container: '#00eefc'
  on-secondary-container: '#00686f'
  tertiary: '#00e55b'
  on-tertiary: '#003911'
  tertiary-container: '#00a740'
  on-tertiary-container: '#00320d'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#ffd9e0'
  primary-fixed-dim: '#ffb1c3'
  on-primary-fixed: '#3f0019'
  on-primary-fixed-variant: '#8f0041'
  secondary-fixed: '#7df4ff'
  secondary-fixed-dim: '#00dbe9'
  on-secondary-fixed: '#002022'
  on-secondary-fixed-variant: '#004f54'
  tertiary-fixed: '#6bff83'
  tertiary-fixed-dim: '#00e55b'
  on-tertiary-fixed: '#002107'
  on-tertiary-fixed-variant: '#00531b'
  background: '#1e1000'
  on-background: '#ffddb4'
  surface-variant: '#4b2f00'
typography:
  headline-xl:
    fontFamily: Space Mono
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 36px
    letterSpacing: -0.04em
  headline-lg:
    fontFamily: Space Mono
    fontSize: 24px
    fontWeight: '700'
    lineHeight: 28px
    letterSpacing: -0.02em
  headline-sm:
    fontFamily: Space Mono
    fontSize: 16px
    fontWeight: '700'
    lineHeight: 20px
    letterSpacing: 0em
  body-lg:
    fontFamily: Space Mono
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0em
  body-sm:
    fontFamily: Space Mono
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0em
  telemetry-display:
    fontFamily: Space Mono
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 28px
    letterSpacing: -0.05em
  telemetry-unit:
    fontFamily: Space Mono
    fontSize: 11px
    fontWeight: '400'
    lineHeight: 12px
    letterSpacing: 0.08em
  label-dense:
    fontFamily: Space Mono
    fontSize: 10px
    fontWeight: '700'
    lineHeight: 12px
    letterSpacing: 0.1em
  code-stream:
    fontFamily: Space Mono
    fontSize: 11px
    fontWeight: '400'
    lineHeight: 14px
    letterSpacing: 0em
spacing:
  gutter: 0.5rem
  gutter-mobile: 0.25rem
  margin: 0.75rem
  margin-mobile: 0.5rem
  space-xs: 0.125rem
  space-sm: 0.25rem
  space-md: 0.5rem
  space-lg: 0.75rem
  space-xl: 1rem
---

## Brand & Style

The design system embodies the raw functional authority of laboratory test benches, modular synthesizers, and real-time energy telemetry consoles. It treats Non-Intrusive Load Monitoring (NILM) as a mission-critical diagnostic stream rather than an abstract dashboard. The user interface directly conveys physical precision: tactile stepped switches, rack-mounted unit compartmentalization, zero diffusion, and uncompromising diagnostic clarity.

The visual style is strictly engineering-grade brutalism merged with industrial telemetry hardware. Surfaces avoid all soft dropshadows, glassmorphism, decorative blurs, or ambient gradient washes. Every container is a hard, discrete physical bay bounded by hairline machine borders. Typography is uniformly monospaced, forcing absolute columnar tracking, numerical alignment, and rigid visual cadences. The operational posture is direct, responsive, high-voltage, and engineered for high-density cognitive throughput under intensive monitoring environments.

## Colors

The color system operates on an ultra-deep, cold-black matte foundation, utilizing specific electromagnetic wavelength accents to classify disaggregated appliance signatures instantly:

- **Base Void (`#090a0f`):** Main application shell background; non-reflective chassis matte finish.
- **Surface Tier 1 (`#0d0e15`):** Structural racks, rail bays, and persistent baseline panel containers.
- **Surface Tier 2 (`#13151f`):** Active card faces, terminal viewports, readout modules, and nested parameter registers.
- **Border Structural Base (`#1e2230`):** Default hairline separation grid lines and module perimeter borders.
- **Border Structural Highlight (`#2a2f42`):** Focused panel frames, active cell demarcations, and perimeter dividers.
- **Primary / Dishwasher Signal (`#ff007a`):** High-voltage electric magenta. Used for mission indicators, primary toggle activations, alert peaks, and inductive load metrics.
- **Secondary / HVAC & Refrigeration (`#00f0ff`):** Electric cyan. Denotes continuous duty cycles, thermodynamic baseline draws, and compressor telemetry.
- **Tertiary / Laundry Cycle (`#00ff66`):** Electric lime. Allocated to dynamic kinetic motor loads, spin phases, and operational clearance states.
- **Auxiliary / Thermal Loads (`#ffaa00`):** Neon amber. Allocated to resistive heating elements, ovens, microwaves, and critical thermal overdraw states.
- **Text & Signal Dim:** `#4a526d` for unit annotations and inactive hardware states; `#8b95b5` for technical parameter labels; `#e1e7f5` for primary telemetry reads.

## Typography

Space Mono serves as the unified typographic engine across all hierarchy tiers. Monospaced rendering ensures absolute vertical parity across stacked data columns, preventing character-width jitter during real-time scalar fluctuations (e.g., kW, RMS Current, THD, Phase Delta).

- All labels and structural metadata must be transformed to uppercase via token implementation (`text-transform: uppercase`).
- Numeric strings must enforce tabular alignment using standard fixed-width figures with zero proportional shifting.
- Letter spacing expands significantly on sub-12px micro-labels (`0.08em` to `0.1em`) to maintain legibility on dark, low-emittance panels, while headlines compress slightly (`-0.02em` to `-0.04em`) to simulate compact physical faceplates.

## Layout & Spacing

The layout is structured as an edge-to-edge instrument modular rack. It follows a dense 16-column fixed-ratio CSS grid architecture designed to maximize horizontal instrumentation space while minimizing inactive margins.

- **Rhythm & Grid:** Built on an absolute 4px base increment. Containers butt against each other through a shared 1px boundary model (`margin: -1px` collision or internal border collapsing) rather than floating island architecture.
- **Adaptive Breakpoints:**
  - **Desktop (>= 1280px):** 16-column matrix layout, split between real-time raw waveforms (minimum 10 columns) and appliance disaggregation channels (6 columns).
  - **Tablet (768px - 1279px):** 8-column layout with pinned top-level instantaneous aggregate telemetry bars and stacked module shelves.
  - **Mobile (< 768px):** 4-column linear vertical rack; oscilloscope traces switch from horizontal continuous scrolling to downsampled interval timebars; all gutters collapse to `0.25rem` (`4px`).

## Elevation & Depth

Visual hierarchy does not rely on elevation z-indexes or dropshadow light sources. The entire system exists on a single structural focal plane simulating a hardware faceplate:

- **Border Containment:** Depth is communicated strictly through surface brightness shifts and crisp 1px borders. A baseline panel `#0d0e15` nests inside `#090a0f`, delimited by a 1px solid `#1e2230` outline.
- **Recessed Insets:** Viewports, spectral analyzers, and data displays simulate countersunk LED arrays using an interior border inset (`box-shadow: inset 0 0 0 1px #090a0f`).
- **Active Energization:** Interactive states do not lift up along a Z-axis. Instead, they ignite: focused borders shift from dormant `#1e2230` to sharp, unblurred `#ff007a`, `#00f0ff`, or `#00ff66`. No glow, bloom, or blur effects are permitted.

## Shapes

The shape system strictly uses 0px roundedness across all boundaries, frames, indicators, badges, and controls.

Corners are absolute, acute right angles (90 degrees). This design reinforces industrial metal stamping, rackmount equipment frontages, and precision laboratory diagnostics. Corner cut-offs (45-degree chamfers measuring precisely 4px) are exclusively permitted for corner hardware registration notches on status matrices, but standard UI elements maintain rigorous `0px` curvature.

## Components

### Telemetry Cards & Module Chassis
- Constructed from `#0d0e15` with a 1px `#1e2230` border.
- Header bars are locked at 24px height, filled with `#13151f`, displaying uppercase 10px monospaced module titles and channel indicators (e.g., `CH_01 // DISHWASHER_HEATING_ELEMENT`).
- Values feature prominent Space Mono bold sizing (`telemetry-display`) paired with a micro-label (`telemetry-unit`) anchored to the baseline.

### Buttons & Hardware Switches
- Buttons feature a 0px radius, 1px solid border, and text set in Space Mono 11px uppercase with `0.1em` letter spacing.
- **Default State:** Background `#13151f`, border `#2a2f42`, text `#8b95b5`.
- **Active/Engaged State:** Inverted solid fill using `#ff007a`, `#00f0ff`, `#00ff66`, or `#ffaa00` with absolute black text (`#090a0f`) and no glow.
- **Trigger Toggle:** Stepped toggle featuring a binary state display `[ ON ] / [ OFF ]` surrounded by a hairline frame.

### Waveform Display & Oscilloscope Containers
- Container uses `#090a0f` background with an integrated 16px CSS-driven grid patterned with `#13151f` lines.
- Waveform line vector paths are razor-sharp 1px or 1.5px lines without ambient anti-aliasing bloom. Color matches the assigned appliance signal accent.

### Chips & Disaggregation Badges
- Compact rectangular tags with 0px radius, 1px border, and 2px horizontal padding.
- Consists of a 6px by 6px solid monochromatic LED block indicator beside raw uppercase status text: `[■ ACTIVE]`, `[■ IDLE]`, `[■ STANDBY]`.

### Checkboxes & Segmented Selectors
- Checkboxes are 12px by 12px hard square frames (`#1e2230` background, `#2a2f42` border). Checked state fills the square with a solid 6px by 6px inner block of `#ff007a` or `#00f0ff`.
- Segmented switches sit flush in a shared housing with zero space between options, separated by internal 1px lines. Selected segments invert to full accent background.

### Input Fields & Parameter Registers
- Flat `#090a0f` background with a continuous `#1e2230` border.
- Text displays in `#e1e7f5` Space Mono. Focus states switch the perimeter border to a 1px solid `#00f0ff` line.
- Left-anchored fixed-width parameter prefix (e.g., `FREQ_HZ >`, `KW_LIMIT >`) rendered in `#4a526d`.