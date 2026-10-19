"""An exposure as hdps sees one: a TANSegment, i.e. a cut-out of an hdps sky tile (0.879"/px TAN) with its WCS.

The engine makes each exposure on a square cut-out of one tile, centred on the observation's block; tile_wcs gives
that cut-out's WCS from hdps's own tiling scheme. Pixel values are electrons in the exposure (sky included), as the
engine makes them; hdps's archived tiles are electrons per second.
"""
import time

import numpy as np
from astropy.io import fits

_SCHEME = None


def tile_wcs(tile_id, ra, dec, origin_x, origin_y):
    """WCS of a cut-out of hdps tile `tile_id` (near ra, dec) whose pixel (0, 0) is tile pixel (origin_x, origin_y)."""
    global _SCHEME
    from hdps.skymap import SegmentationScheme

    if _SCHEME is None:
        _SCHEME = SegmentationScheme.all_sky()
    wcs = dict(_SCHEME.query_tile_wcses_with_ids(ra - 0.05, dec - 0.05, ra + 0.05, dec + 0.05))[tile_id].deepcopy()
    wcs.wcs.crpix = wcs.wcs.crpix - np.array([origin_x, origin_y])
    return wcs


def make_frame(image, wcs, tile_id, metadata):
    """An hdps TANSegment holding one exposure (metadata: band, exposure, time, exptime, airmass, seeing, ...)."""
    from hdps.core.images import TANSegment
    from hdps.core.telemetry import Telemetry

    ra, dec = wcs.wcs_pix2world([[image.shape[1] / 2, image.shape[0] / 2]], 0)[0]
    return TANSegment(data=np.asarray(image, np.float32), metadata=dict(metadata), timestamp=time.time(),
                      telemetry=Telemetry(output_enabled=False), full_wcs=wcs, linear_wcs=wcs, tile_id=int(tile_id),
                      ra_center=float(ra), dec_center=float(dec))


def write(frame, path):
    """The frame as FITS: the image with its WCS and metadata in the header."""
    header = frame.full_wcs.to_header()
    header["TILE_ID"] = frame.tile_id
    for key, value in frame.metadata.items():
        header[key[:8].upper()] = value
    fits.writeto(path, frame.data, header, overwrite=True)
