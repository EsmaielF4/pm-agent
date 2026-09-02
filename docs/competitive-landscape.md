# Competitive Landscape — Predictive Maintenance in Iran

Research compiled for the 100RFS pitch. Three categories of existing
players, plus one critical fact: the organization hosting this
competition already runs internal predictive-maintenance infrastructure.

## 1. AI-native startups — closest direct competitor

### SeeNous (seenous.com)
- HQ: Mashhad Innovation Factory, Iran. 11–50 employees (per public
  company data). CEO: Mehdi Zarkak.
- Product: remote condition-monitoring platform combining AI with
  human experts, focused on vibration-based failure diagnosis for
  rotating equipment. Also suggests the corrective action and the best
  timing to perform it, not just a fault flag.
- Features: cloud access to equipment data collected remotely, 3D
  models of equipment for visual context, threshold-based flagging,
  trend visualization, integration with equipment history.
- Has at least one named customer testimonial (APSA Company), citing
  high accuracy in rotating-equipment failure detection.
- Explicitly states it operates globally, not just in Iran.
- **This is the closest existing product to what pm_agent could become.**
  Their apparent focus is a single specialized capability (vibration
  diagnosis). Our architecture's differentiator is proven expandability
  across structurally different task types (binary fault detection,
  multi-label diagnosis, RUL regression) on one plugin system — not
  just one diagnostic capability.

## 2. Established CMMS vendors — added AI as a feature, not built AI-first

### PMworks (pmworks.ir)
- Mature, scaled business: 240+ client organizations, ISO 9001 and
  IATF 16949 (automotive industry) standard compliance.
- Core product: 16-module web-based CMMS (work orders, PM scheduling,
  failure analysis, spare-parts management, KPI reporting), works for
  large and small industries, on-premise or web-hosted.
- Newer cloud tier adds an AI layer (expanding to ~27 modules) that
  suggests failure modes and maintenance activities.
- No public pricing — sales-contact/demo-request model.
- **Positioning note**: this is a workflow/CMMS incumbent that bolted
  on AI, not an AI-first predictive engine. They already own the
  work-order relationship with 240+ plants — a real distribution
  advantage we don't have.

### Pegah Aftab (pegaheaftab.com)
- Broader enterprise software house: HR management ("Tabaan"), process/
  workflow management ("Afrooz"), e-office ("Afaq"), plus a maintenance
  module ("Net Pegah").
- Net Pegah is explicitly a **preventive**-maintenance (PM) tool —
  weekly scheduling, spare-parts allocation, reporting — not
  predictive/AI-based, based on their own site content.
- Client base includes several named Iranian power plants (visible via
  logos on their homepage).
- **Positioning note**: purely a scheduled-maintenance workflow tool,
  no predictive/AI capability found. Furthest from our space, but
  relevant as an incumbent with existing power-plant relationships.

## 3. Traditional vibration-analysis / condition-monitoring consultancies

The existing, non-AI, expert-driven way this job gets done today in
Iran — useful as the "status quo we're improving on" framing:

- **Noavaran Payesh** — equipment + services: vibration/balance
  measurement, shaft alignment, thermography, online vibration sensors.
- **ABP Vibro (Behine Pardazesh Arman)** — knowledge-based company
  ("دانش‌بنیان"), 19+ years designing/building vibration-measurement
  equipment for rotating machinery.
- **Vista Payesh Rad** — vibration-analysis condition-monitoring
  services.
- **Akopayesh** — vibration analysis / condition-monitoring consulting.
- **Davar Machine Pars** — vibration analysis with an explicit CBM
  roadmap methodology: routine data collection, analysis, repair
  prioritization, periodic technical reports.

These represent expert-led, periodic, largely manual condition
monitoring — real domain expertise, but not continuously-learning AI.

## 4. MAPNA's own internal predictive-maintenance infrastructure

**The single most important competitive fact**: MAPNA (the host of
this competition) already operates real predictive-maintenance/
condition-monitoring infrastructure internally.

- **MAPNA Digital** operates under the "Mapna Mind" brand, offering an
  integrated predictive-maintenance software platform, AI-based
  condition monitoring, and integration with CMMS/APM systems — as
  part of MAPNA's service offering to its own plant-service clients.
- **MECO** (Mapna Electric Control Engineering & Manufacturing, a
  MAPNA group company) built and has operated an online Remote
  Diagnostic Center (RDC) since 2017, remotely monitoring MAPNA's own
  thermal and wind power plant fleet — gas turbines, wind turbines,
  generators, transformers.
- A MAPNA-affiliated power plant recently opened an AI-based smart
  condition-monitoring center built in close collaboration with
  **Sharif University**, reporting a **10% reduction in forced outage
  hours (FOH)** from the predictive capability.

### What this means for the pitch
- **It validates the problem strongly** — MAPNA has already invested
  real money and years into this internally, with a university
  research partner, and can point to concrete measured results.
- **The honest gap to point to**: MAPNA's internal infrastructure
  (RDC, Mapna Mind) appears built specifically for MAPNA's own
  large-scale power-generation fleet (turbines, generators,
  transformers). The MAPNA 100RFS dataset (P2–P5) explicitly covers
  pumps and general rotary equipment — a different, much larger
  category used across oil & gas, petrochemical, and general industry,
  far beyond MAPNA's own power plants.
- **A credible answer to "why you, when MAPNA already has this"**:
  MAPNA's internal system was purpose-built for MAPNA's own large
  fleet with a university research partnership behind it. The
  underserved market is the long tail of smaller industrial plants and
  non-power-generation rotary equipment that will never get a bespoke,
  internally-funded build like that — that's the gap an external,
  expandable, lower-cost solution could fill.

## Summary positioning

| Competitor | AI-native? | Task breadth | Target |
|---|---|---|---|
| SeeNous | Yes | Single (vibration diagnosis) | Rotating equipment, global |
| PMworks | AI added on | CMMS + AI suggestions | 240+ orgs, all industries |
| Pegah Aftab | No | Preventive scheduling only | Power plants (named clients) |
| Vibration consultancies | No | Manual expert analysis | Rotating equipment |
| MAPNA internal (RDC/Mapna Mind) | Yes | Turbines/generators/transformers | MAPNA's own large fleet |
| **pm_agent (this project)** | Yes | 3 proven task types, expandable architecture | Pumps + general rotary equipment, broader industrial market |