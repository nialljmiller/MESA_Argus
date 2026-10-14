# Night movies: what is modelled, what is taken from ArgusSim, and what is assumed

This applies to every observation made by `observe.py` (and the hand-made fields M3, M15 and two RR Lyrae fields).
Stars are simulated by `make_stars.py` with no dust and nothing about observing; each built star's own choices are in
`stars/<name>/summary.md`, and each observation's (the night, background, placements, dust) in
`outputs/<name>/summary.md` and `observation.toml`.

## 1. Argus, taken directly from ArgusSim (not chosen here)

| What | Source in ArgusSim |
|---|---|
| β and ρ throughput: filter × optics × QE, throughput loss | `SystemThroughput` (config defaults) |
| Collecting area | `config.telescope.collecting_area` |
| Plate scale 1.007″/px, full well 21,657 e⁻, read noise 1.2 e⁻, dark 0.0022 e⁻/s/px | `config.telescope` |
| PSF: Zemax optical PSF with the delivered EE50 profile, convolved with seeing, block-summed to pixels | `ZemaxPSF.from_config(...).with_seeing(...)` |
| Atmospheric transmission, raised to the airmass | `SystemThroughput` atmosphere curve; `ABPhot.electrons_from_sed` |
| Sky per exposure: airglow (solar activity, airmass), zodiacal light (ecliptic position), integrated starlight (Galactic latitude), moonlight (Krisciunas & Schaefer 1991), site artificial light | `Observatory.sky_components_at` + `SpectralSky.sky_electrons` |
| Seeing: the site's seasonal median from the DIMMs, scaled to airmass as X^0.6 | `seasonal_seeing_params`, `seeing_at_airmass` |
| The night's exposures: 60 s cadence in 15-min ratchets on UTC quarter hours, the last minute a reset, the Sun below −18° | `coverage_exact.Schedule` built from `config.survey` (as `Survey` builds it) |
| When a field can be observed: at or above the array's minimum altitude, 32.84° (airmass ≈ 1.84) | `config.observatory.min_alt` |
| Airmass: Pickering (2002), for the stars' extinction and seeing as for the sky | `pickering_airmass` |
| Site | `config.observatory` (a field may move it, e.g. `--site sro`; the instrument stays Argus) |
| Synthetic AB magnitudes (filter only) and detected electrons (full system) from an SED | `ABPhot.mag_from_sed`, `ABPhot.electrons_from_sed` |
| HDPS minipix geometry (32 × 0.879″ = 28.1″) | `skymap_shim` |

## 2. Written here (ArgusSim has no equivalent): images and photometry

ArgusSim supplies one star's PSF at a time (`ZemaxPSF.with_seeing`) and an analytic S/N (`NoiseBudget.get_snr`). It
does not make images or measure them. The following is this repo's own code, built on ArgusSim's inputs.

- **Scene** (`night_movie.Scene`). Every source's position becomes a pixel position. Its ArgusSim PSF is taken at
  that sub-pixel position (rounded to 1/8 pixel), scaled by its ArgusSim electrons for the exposure (its SED at that
  moment, at that airmass), and summed with ArgusSim's sky and dark current.
- **Noise.** A Poisson draw per pixel, clipped at ArgusSim's full well, plus Gaussian read noise.
- **Photometry** (`mesa_argus/photometry.py`). SExtractor (its Python port SEP) with SExtractor's defaults on every
  exposure: background mesh, 1.5σ detection, deblending, windowed centroids, MAG_AUTO. The real Argus pipeline does
  no PSF or aperture photometry of its own, so a standard extractor stands in for it.
- **Calibration.** Each exposure separately: about 300 bright, least-blended catalogue stars give the zero point with
  a linear colour term in BP−RP (3σ clipped). A source without a catalogue colour (every transient) takes the median
  colour of the calibration stars. A source much bluer or redder than that median is therefore measured with a
  colour-term bias: about −0.17 mag for a blue SN Ia, about −0.05 to −0.1 mag for a dwarf-nova disc.
- **Difference image** (display only). The reference is the median of the night's best-seeing exposures (each a full
  noisy simulated exposure; the transients left out, i.e. the field before the event). It is PSF-matched to each
  exposure with a Bramich (2008) delta-function kernel, fitted on stamps around reference sources with 3σ rejection.
- **Colour exposure** (display only). β and ρ, sky subtracted; ρ → red, β → blue, their mean → green, with β scaled
  so the field's median star is grey. Brightness is ρ electrons per pixel above sky on an asinh stretch; colour is the
  pixel's channel ratios (Lupton et al. 2004); a pixel at full well in either band is white.

## 3. Argus-related choices made here (assumptions, not from ArgusSim)

Items 1–5 are conditions ArgusSim's own survey simulation draws (Survey, ForwardMC) and these movies do not use yet.

1. **Field angle 1.44°, fixed all night.** The area-weighted median of the 2.03° field. The PSF depends strongly on
   it (HFD 2.34″ at 1°, 2.70″ at 1.44°, 3.25″ at 2°). ArgusSim's survey takes it per ratchet (`random_field_angle`,
   or exact from `coverage_exact`).
2. **Continuous coverage, both bands.** Every exposure of the night is taken to image the field in both β and ρ with
   one telescope. ArgusSim's `coverage_exact` / `tanseg_coverage` say which OTA (so which single band) covers a
   position in each exposure, and when it falls in a gap.
3. **A median-seeing night.** The seasonal median zenith seeing for the whole night, varying only with airmass.
   ArgusSim's survey draws it per ratchet (`draw_atmosphere`: lognormal about the seasonal median, correlated with
   transparency).
4. **Photometric sky, every night clear.** Transparency 1. ArgusSim draws a grey cloud transparency per exposure
   (`draw_cloud_transparency`) and whether a night is clear (`observable_weather`).
5. **Solar activity 0.8** (ArgusSim's config default, solar minimum), which sets the airglow level. ArgusSim's survey
   uses a per-night 2028–2033 prior (`solar_activity_by_ratchet`).
6. **PSF details.** The PSF stamp ends at 29 × 29 pixels (ArgusSim's Zemax grid limit), so very bright stars lose
   their far wings. Detector axes are taken as aligned with the HDPS tile axes. The same seeing FWHM is used in β and
   ρ (real seeing scales as λ^−0.2, so β's would be ~8% broader; ArgusSim has no chromatic seeing either).
7. **Not modelled at all:** flat-field and calibration errors, detector non-linearity, blooming beyond full-well
   clipping, cosmic rays, persistence, gain and ADC quantisation, scintillation, tracking jitter and image rotation.

## 4. Astrophysics: modelled vs assumed

Every source is an SED; ArgusSim turns it into electrons in each band at each exposure's airmass. Milky Way dust is
CCM89 (R_V = 3.1; `mesa_argus/dust.py`), added at observation, with E(B−V) the dust in front of each star: Bayestar19
(Green et al. 2019) at the star's distance along its sightline, never more than SFD's whole column (beyond a
sightline's reliable distances Bayestar19 follows its prior); SFD where Bayestar19 has no data (dec < −30°) or the star
has no distance (supernovae, microlensing sources: behind all the dust). E(B−V) = 0.884 × the map value (both maps are
in SFD units; with CCM89 this gives Schlafly & Finkbeiner's 2011 A_V = 2.742 per SFD unit). Distances: real stars,
Bailer-Jones et al. (2021) geometric, else Gaia GSP-Phot, else none (the whole column); Gaia variables, their own
(parallax or period-luminosity, iterated with the dust); placed periodic stars and dwarf novae, their model's.

- **Constant stars.** Gaia DR3 measured each through the dust in front of it; that dust is taken out with the same
  law: the MESA SED (the `argus_sim_seds` tracks: 0.8 M☉ [M/H] = −2; 1, 3, 5 and 10 M☉ solar) whose BP−RP, seen
  through that E(B−V), is the measured one gives the intrinsic colour, and its A_G the dust-free G. At observation that
  SED, scaled to the dust-free G, is seen through the dust where the star is observed: at its real position, exactly as
  Gaia measured it; in a synthetic field, through the new sightline. Only colour is matched (not gravity, metallicity
  or mass). Above E(B−V) ≈ 0.9 neighbouring templates of other metallicities redden slightly differently (≤ 0.05 mag
  in BP−RP at E(B−V) = 4); the inversion uses their running maximum. Sources with no or blended BP−RP take a median
  colour.
- **Pulsators** (`mesa_argus/rsp.py`): MESA RSP models whose linear period of the star's own mode is tuned to the
  Gaia (or ASAS-SN) period, run to the limit cycle, then one cycle with an SED every ≤ 25 s from MESA's colors module.
  Placed on the night with the catalogue period and epoch, so the timing of each maximum is a prediction with the
  catalogue's propagated uncertainty. Amplitudes and shapes are the models' predictions, not fitted. Mass assumptions
  by class: RR Lyrae from [M/H]; classical Cepheids from the Bono et al. (2000) mass–luminosity relation; Type II
  Cepheids 0.55 M☉; HADS from the Gaia FLAME mass, radiative (convection off).
- **Eclipsing binaries** (`mesa_argus/binary.py`): PHOEBE 2 with MESA-atmosphere SEDs, timed and shaped by Gaia's
  eclipsing-binary model (and its NSS orbit when it has one); masses from a main-sequence relation; scaled to Gaia's
  out-of-eclipse G.
- **Stars made from parameters** (`make_stars.py`; `mesa_argus/sources/`, `python -m mesa_argus.sources`): defined
  entirely by the star file's keys, with no Gaia input and no dust.
  - Periodic: MESA RSP pulsators (RR Lyrae, Cepheids, Type II Cepheids, HADS) tuned to the given period, at the given L
    (or dust-free absolute M_G) and distance; PHOEBE binaries (W UMa from period, q, inclination, Teffs and fillout
    factor; detached from radii and Teffs) at the given distance, masses from the main-sequence Teff–mass relation
    unless given. Placed by an observation (position, epoch), they enter the catalogue (and the difference reference)
    at their intensity-mean G and BP−RP through the field's dust, as a catalogued variable would.
- **Transients** (`mesa_argus/transients.py`): SEDs against time since t0.
  - sncosmo models (supernova templates built from observed spectra; no physics run), placed at a redshift and peak
    absolute magnitude with sncosmo's default cosmology.
  - Two parametric models written in this repo, not from sncosmo: point-lens microlensing (MESA template source), and
    a dwarf-nova outburst (steady accretion disc
    of changing accretion rate plus the white dwarf; no donor, bright spot or lines).
  - Transients are not in the catalogue (no catalogue colour; see calibration above) and not in the difference
    reference.
- **Variables not modelled** (held constant at their Gaia mean): double-mode pulsators, non-radial pulsators,
  long-period, rotational, eruptive and stochastic variables, and any star whose model failed to reproduce its
  catalogue period or eclipses. `make_stars.py --gaia-list RA DEC` gives the reason per star.
- **Synthetic fields** (`[stars] layout = "random"`): every star of a Gaia field set, placed uniformly at random
  within 0.12 deg of the pointing at its own distance, its own dust taken out and the dust in front of it at the new
  position put in; every star constant; the pointing's night and sky.
- **Positions.** Gaia DR3 (fine for one night). Gaia misses faint stars in crowded fields, so unresolved light is
  underestimated there.

### The original field: M3

The first movie (`--field m3`, night 2027-05-05, new Moon) follows four RRab and two RRc in an 84″ block of M3's core
(about 1,550 Gaia sources). Their RSP models use X = 0.75, Z = 0.001, α-enhanced (a09) opacities, 0.68 M☉ and
convection set A. Matching each period with only that mode unstable needed L lowered from M3's horizontal-branch
level (~50 L☉) by 20% for the RRab and 5–10% for the RRc, so the RRab models are ~0.25 mag fainter than the real
stars and their ρ amplitudes ~35% smaller. M3 is at 10.18 kpc (Baumgardt & Vasiliev 2021) with E(B−V) = 0.01
(Harris). The block's 15.3 d Type II Cepheid could not be modelled (every RSP setup crashed or diverged) and is
constant. In the core, 1,424 of 2,702 sources have missing or blended BP−RP and take the median colour of outer-M3
stars at the same G.

## 5. What the panels show

- **CMD:** grey, the constant stars at their true (template) magnitudes; coloured tracks, each followed source's model
  (one cycle, or the night for a transient); coloured points, its **true** model value now, not a measurement.
- **Exposure:** the β/ρ colour composite of this exposure (section 2).
- **Difference:** this exposure minus the PSF-matched reference, in electrons per pixel (saturated pixels grey).
- **Light curves:** MAG_AUTO measured on each exposure (points) and the model (line), relative to the night's mean;
  below, measured − true.
- **SED:** each followed source's SED now (solid), and the same SED times the β and ρ system responses at this
  airmass (dotted).
