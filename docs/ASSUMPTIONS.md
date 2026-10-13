# Night movies: what is modelled, what is taken from ArgusSim, and what is assumed

This applies to every movie made by `night_movie.py`, whether `argus_field.py` set up the field (Gaia or synthetic)
or it is one of the hand-made fields (M3, M15, two RR Lyrae fields). Each field's own choices are listed in
`outputs/<field>/field_summary.md`: which stars, which models, the night, distance and reddening.

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
| Cadence: 60 s exposures; 15-min ratchets on UTC quarter hours, the last minute a reset | `config.survey` |
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

1. **Field angle 1.44°, fixed all night.** The area-weighted median of the 2.03° field. The PSF depends strongly on
   it (HFD 2.34″ at 1°, 2.70″ at 1.44°, 3.25″ at 2°). In reality a field lands at a different place on the detector
   each ratchet, and possibly on different telescope units.
2. **Continuous coverage.** Every 60 s exposure of the night (all 14 per ratchet) is assumed to image the field with
   one telescope. Not checked against ArgusSim's ratchet/coverage engine.
3. **A median-seeing night.** The seasonal median, with no nightly draw and no variation apart from airmass.
4. **Photometric sky.** Transparency 1 (no cloud); weather not drawn.
5. **Solar activity 0.8** (ArgusSim's config default, solar minimum). This sets the airglow level.
6. **PSF details.** The PSF stamp ends at 29 × 29 pixels (ArgusSim's Zemax grid limit), so very bright stars lose
   their far wings. Detector axes are taken as aligned with the HDPS tile axes.
7. **Not modelled at all:** flat-field and calibration errors, detector non-linearity, blooming beyond full-well
   clipping, cosmic rays, persistence, gain and ADC quantisation, scintillation, tracking jitter and image rotation.

## 4. Astrophysics: modelled vs assumed

Every source is an SED; ArgusSim turns it into electrons in each band at each exposure's airmass. Milky Way dust is
CCM89 (`mesa_argus/dust.py`) with the field's E(B−V) (SFD at the centre for `argus_field.py` fields).

- **Constant stars.** Brightness: Gaia DR3 G. SED: the MESA SED (the `argus_sim_seds` tracks: 0.8 M☉ [M/H] = −2;
  1, 3, 5 and 10 M☉ solar) whose Gaia BP−RP is closest, scaled to G. Only colour is matched (not gravity, metallicity
  or mass). Their fluxes include reddening through their observed Gaia magnitudes. Sources with no or blended BP−RP
  take a median colour.
- **Pulsators** (`mesa_argus/rsp.py`): MESA RSP models whose linear period of the star's own mode is tuned to the
  Gaia (or ASAS-SN) period, run to the limit cycle, then one cycle with an SED every ≤ 25 s from MESA's colors module.
  Placed on the night with the catalogue period and epoch, so the timing of each maximum is a prediction with the
  catalogue's propagated uncertainty. Amplitudes and shapes are the models' predictions, not fitted. Mass assumptions
  by class: RR Lyrae from [M/H]; classical Cepheids from the Bono et al. (2000) mass–luminosity relation; Type II
  Cepheids 0.55 M☉; HADS from the Gaia FLAME mass, radiative (convection off).
- **Eclipsing binaries** (`mesa_argus/binary.py`): PHOEBE 2 with MESA-atmosphere SEDs, timed and shaped by Gaia's
  eclipsing-binary model (and its NSS orbit when it has one); masses from a main-sequence relation; scaled to Gaia's
  out-of-eclipse G.
- **Transients** (`mesa_argus/transients.py`): SEDs against time since t0.
  - sncosmo models (supernova templates built from observed spectra; no physics run), placed at a redshift and peak
    absolute magnitude with sncosmo's default cosmology.
  - Three parametric models written in this repo, not from sncosmo: a GRB afterglow (power law in time and frequency
    with a t² onset), point-lens microlensing (MESA template source), and a dwarf-nova outburst (steady accretion disc
    of changing accretion rate plus the white dwarf; no donor, bright spot or lines).
  - Transients are not in the catalogue (no catalogue colour; see calibration above) and not in the difference
    reference.
- **Variables not modelled** (held constant at their Gaia mean): double-mode pulsators, non-radial pulsators,
  long-period, rotational, eruptive and stochastic variables, and any star whose model failed to reproduce its
  catalogue period or eclipses. `argus_field.py --list` gives the reason per star.
- **Synthetic fields** (`argus_field.py --synthetic`): the G, BP and RP of every Gaia DR3 source around the real
  position, placed uniformly at random in the same circle; every star constant; the real position's night, sky and
  reddening.
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
