"""An exposure on a telescope's sensor, as ArgusSim defines the array: where each star falls, and the image.

Geometry (all ArgusSim, argus_sim.coverage_exact): a position goes to the array frame as covering(apparent=True) puts
it (ICRS -> CIRS once per night at the night's mean exposure time; array-frame hour angle = Earth rotation angle at the
site + the exposure's tracking angle theta - RA; alt/az at the site latitude), then into the telescope's focal plane
as covering's _members tests membership: gnomonic coordinates along the camera's long axis (et, along the ring) and
short axis (er, radial) of Layout.basis. Sensor pixel = the sensor centre + the gnomonic coordinate x focal length /
pixel size (ArgusSim telescope.focal_length, pixel_size, n_pixels_x, n_pixels_y): x along the long axis, y along the
short. The field angle (the PSF depends on it) is the angle from the telescope's axis.

The image (ours: ArgusSim has no image renderer): each star's ArgusSim ZemaxPSF stamp at its field angle and the
exposure's seeing, at its sub-pixel phase, scaled by its electrons; plus sky + dark per pixel; Poisson; read noise;
clipped at the full well.
"""
import numpy as np

PHASES = 4  # sub-pixel phases per axis for the PSF stamps (as night_movie's PSF_PHASES)


def array_frame(ra, dec, times):
    """(alt, az) in ArgusSim's array frame for positions ra, dec (deg) at each time, before the tracking angle:
    returns a function of theta_deg -> (alt, az) arrays of shape (n_times, n_positions)."""
    import astropy.units as u
    from argus_sim import c
    from argus_sim.coverage_exact import _hadec_to_altaz
    from astropy.coordinates import CIRS, SkyCoord
    from astropy.time import Time

    times = Time(times)
    nights = np.floor(times.mjd + c.observatory.longitude / 360.0 - 0.5)
    ra_c = np.empty((len(times), len(ra)))
    dec_c = np.empty_like(ra_c)
    for n in np.unique(nights):  # ICRS -> CIRS once per night, at the night's mean time (covering, apparent=True)
        these = nights == n
        mid = Time(np.mean(times.mjd[these]), format="mjd", scale="utc")
        cirs = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg).transform_to(CIRS(obstime=mid))
        ra_c[these], dec_c[these] = cirs.ra.deg, cirs.dec.deg
    era = times.earth_rotation_angle(longitude=c.observatory.longitude * u.deg).deg

    def at(theta_deg):
        ha = ((era + np.asarray(theta_deg))[:, None] - ra_c + 180.0) % 360.0 - 180.0
        alt, az = _hadec_to_altaz(ha, dec_c, c.observatory.latitude, np)
        return alt, az

    return at


def focal_plane(alt, az, ota, layout):
    """Gnomonic coordinates (u along the long axis, v along the short) in telescope `ota`'s focal plane, and the
    field angle (deg), for array-frame positions alt, az (any shape)."""
    from argus_sim.coverage_exact import _unit

    C, et, er = layout.basis()
    P = _unit(np.ravel(alt), np.ravel(az), np)
    d = P @ C[ota]
    u = (P @ et[ota]) / d
    v = (P @ er[ota]) / d
    angle = np.degrees(np.arccos(np.clip(d, -1.0, 1.0)))
    shape = np.shape(alt)
    return u.reshape(shape), v.reshape(shape), angle.reshape(shape)


def to_pixels(u, v):
    """Sensor pixel (x along the long axis, y along the short; 0-based, centres at integers) of gnomonic u, v; the
    camera's readout orientation (camera.READOUT_Y_SIGN: ArgusSim does not define it) sets the direction of y."""
    from argus_sim import c

    from mesa_argus import camera

    t = c.telescope
    f_px = t.focal_length / (t.pixel_size / 1000.0)
    return (t.n_pixels_x - 1) / 2.0 + u * f_px, (t.n_pixels_y - 1) / 2.0 + camera.READOUT_Y_SIGN * v * f_px


def sky_at(x, y, time, theta_deg, ota, layout):
    """(RA, Dec) ICRS (deg) at sensor pixel (x, y) in an exposure: the inverse of array_frame -> focal_plane ->
    to_pixels (ArgusSim's geometry; CIRS -> ICRS at the exposure time)."""
    import astropy.units as u
    from argus_sim import c
    from astropy.coordinates import CIRS, SkyCoord
    from astropy.time import Time

    from mesa_argus import camera

    t = c.telescope
    f_px = t.focal_length / (t.pixel_size / 1000.0)
    gu = (np.asarray(x, float) - (t.n_pixels_x - 1) / 2.0) / f_px
    gv = (np.asarray(y, float) - (t.n_pixels_y - 1) / 2.0) / (camera.READOUT_Y_SIGN * f_px)
    C, et, er = layout.basis()
    P = C[ota] + gu[..., None] * et[ota] + gv[..., None] * er[ota]
    P = P / np.linalg.norm(P, axis=-1, keepdims=True)
    alt, az = np.arcsin(np.clip(P[..., 2], -1, 1)), np.arctan2(P[..., 1], P[..., 0])
    lat = np.radians(c.observatory.latitude)
    dec = np.arcsin(np.clip(np.sin(lat) * np.sin(alt) + np.cos(lat) * np.cos(alt) * np.cos(az), -1, 1))
    ha = np.degrees(np.arctan2(-np.cos(alt) * np.sin(az), np.sin(alt) * np.cos(lat) - np.cos(alt) * np.sin(lat) * np.cos(az)))
    when = Time(time)
    era = when.earth_rotation_angle(longitude=c.observatory.longitude * u.deg).deg
    ra_cirs = (era + float(theta_deg) - ha) % 360.0
    icrs = SkyCoord(ra_cirs * u.deg, np.degrees(dec) * u.deg, frame=CIRS(obstime=when)).icrs
    return icrs.ra.deg, icrs.dec.deg


def window(x0, y0, side_px):
    """The readout window (x_origin, y_origin, width, height) of side side_px centred on sensor pixel (x0, y0) and
    kept on the sensor; side_px 0: the whole sensor."""
    from argus_sim import c

    nx, ny = c.telescope.n_pixels_x, c.telescope.n_pixels_y
    if not side_px:
        return 0, 0, nx, ny
    side = int(min(side_px, nx, ny))
    xo = int(np.clip(round(x0 - side / 2), 0, nx - side))
    yo = int(np.clip(round(y0 - side / 2), 0, ny - side))
    return xo, yo, side, side


class PSFs:
    """ArgusSim ZemaxPSF stamps (unit flux) by (band, seeing, tabulated field angle, sub-pixel phase), made as
    needed and kept."""

    def __init__(self, cfg):
        from argus_sim import ZemaxPSF

        self.model = ZemaxPSF.from_config(cfg)
        self.cache = {}

    def angles(self, band):
        from mesa_argus.argus import BANDS

        return self.model.field_angles(BANDS.get(band, band))  # ArgusSim's band names (beta: "g")

    def stamp(self, band, seeing, angle_index, py, px):
        key = (band, round(float(seeing), 3), int(angle_index), int(py), int(px))
        if key not in self.cache:
            angle = self.angles(band)[angle_index]
            from mesa_argus.argus import BANDS

            p, _ = self.model.with_seeing(seeing=float(seeing), band=BANDS.get(band, band), field_angle_deg=float(angle),
                                          pixel_phase=(py / PHASES, px / PHASES))
            p = np.maximum(p, 0.0)
            self.cache[key] = (p / p.sum()).astype(np.float32)
        return self.cache[key]


def render(x, y, electrons, field_angle, band, seeing, background, win, psfs, rng, readnoise, fullwell):
    """One exposure in the readout window win = (x_origin, y_origin, width, height): stars at sensor pixels x, y with
    `electrons` each (their PSFs at their field angles), plus `background` electrons per pixel (sky + dark); Poisson,
    read noise, clipped at the full well. Returns the image (electrons, float32, height x width)."""
    xo, yo, w, h = win
    mean = np.full((h, w), float(background), np.float32)
    angles = psfs.angles(band)
    ai = np.abs(field_angle[:, None] - angles[None, :]).argmin(axis=1)  # nearest tabulated angle (as ZemaxPSF)
    lx, ly = x - xo, y - yo
    # each star's stamp origin and sub-pixel phase, so the stamp's centroid lands on the star (the stamp centroid
    # moves by -phase; as night_movie's Scene)
    s0 = psfs.stamp(band, seeing, 0, 0, 0)
    yy, xx = np.indices(s0.shape)
    cy, cx = float((s0 * yy).sum()), float((s0 * xx).sum())
    n0 = s0.shape[0]
    inside = (lx > -n0) & (lx < w + n0) & (ly > -n0) & (ly < h + n0) & (electrons > 0)
    for k in np.nonzero(inside)[0]:
        ox, oy = int(np.ceil(lx[k] - cx)), int(np.ceil(ly[k] - cy))
        px = int(np.clip(round((ox + cx - lx[k]) * PHASES), 0, PHASES - 1))
        py = int(np.clip(round((oy + cy - ly[k]) * PHASES), 0, PHASES - 1))
        p = psfs.stamp(band, seeing, ai[k], py, px)
        n = p.shape[0]
        x0, x1, y0, y1 = max(ox, 0), min(ox + n, w), max(oy, 0), min(oy + n, h)
        if x1 > x0 and y1 > y0:
            mean[y0:y1, x0:x1] += electrons[k] * p[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    image = rng.poisson(mean).astype(np.float32)
    np.minimum(image, fullwell, out=image)
    image += rng.normal(0.0, readnoise, image.shape).astype(np.float32)
    return image


class Projector:
    """Every star's sensor position in every exposure, as ArgusSim's geometry places it (array_frame -> focal_plane
    -> to_pixels): ICRS -> CIRS once per night (at the night's mean exposure time, as covering), then per exposure
    the array-frame transform with its tracking angle and its telescope's axes."""

    def __init__(self, ra, dec, times, layout):
        import astropy.units as u
        from argus_sim import c
        from astropy.coordinates import CIRS, SkyCoord
        from astropy.time import Time

        self.layout = layout
        self.times = Time(times)
        nights = np.floor(self.times.mjd + c.observatory.longitude / 360.0 - 0.5)
        self.night = np.unique(nights, return_inverse=True)[1]
        self.cirs = []
        for n in range(self.night.max() + 1):
            mid = Time(np.mean(self.times.mjd[self.night == n]), format="mjd", scale="utc")
            cirs = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg).transform_to(CIRS(obstime=mid))
            self.cirs.append((cirs.ra.deg, cirs.dec.deg))
        self.era = self.times.earth_rotation_angle(longitude=c.observatory.longitude * u.deg).deg

    def pixels(self, j, theta_deg, ota):
        """Sensor x, y (px) and field angle (deg) of every star in exposure j."""
        from argus_sim import c
        from argus_sim.coverage_exact import _hadec_to_altaz

        ra_c, dec_c = self.cirs[self.night[j]]
        ha = (self.era[j] + float(theta_deg) - ra_c + 180.0) % 360.0 - 180.0
        alt, az = _hadec_to_altaz(ha, dec_c, c.observatory.latitude, np)
        u, v, angle = focal_plane(alt, az, int(ota), self.layout)
        x, y = to_pixels(u, v)
        return x, y, angle
