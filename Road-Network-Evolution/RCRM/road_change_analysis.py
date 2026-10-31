"""Road Change Characterisation -> RCRM -> Priority Analysis (slides 21-27).

Inputs are binary road masks for T1/T2 (from CRNet) and the change
probability map (from LGMM-Net). Nothing here needs a GPU.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

_EIGHT = np.ones((3, 3), bool)


def _dilate(mask, r):
    if r <= 0:
        return mask.copy()
    return ndi.binary_dilation(mask, structure=ndi.generate_binary_structure(2, 2), iterations=r)


def _skeleton_nodes(skel):
    """Junction + endpoint pixels of a 1-px skeleton."""
    nb = ndi.convolve(skel.astype(np.uint8), _EIGHT.astype(np.uint8), mode="constant") - skel
    return skel & ((nb == 1) | (nb >= 3))


@dataclass
class Analysis:
    regions: list = field(default_factory=list)   # list of dicts, sorted by priority score (desc)
    labels: np.ndarray = None                      # int map, region id per pixel (0 = none)
    change_mask: np.ndarray = None


def analyse(roads_t1, roads_t2, change_prob, cfg):
    """Characterise every change region and score it."""
    roads_t1 = roads_t1.astype(bool)
    roads_t2 = roads_t2.astype(bool)
    union = roads_t1 | roads_t2
    change = change_prob >= cfg.change_threshold

    comp, n = ndi.label(change, structure=_EIGHT)
    union_buf = _dilate(union, cfg.road_buffer_px)

    # network skeleton, its nodes, and which connected network piece each pixel is in
    skel = skeletonize(union)
    nodes = _skeleton_nodes(skel)
    node_lab, _ = ndi.label(_dilate(nodes, 1), structure=_EIGHT)  # merge neighbouring node pixels
    node_lab = np.where(nodes | _dilate(nodes, 1), node_lab, 0)
    net_lab, _ = ndi.label(skel, structure=_EIGHT)
    # which network piece each node belongs to (computed once, reused per region)
    nodes_of_net = {}
    ys, xs = np.nonzero(nodes)
    for nl, nd in zip(net_lab[ys, xs].tolist(), node_lab[ys, xs].tolist()):
        if nl and nd:
            nodes_of_net.setdefault(nl, set()).add(nd)

    out = []
    kept = np.zeros_like(comp)
    objs = ndi.find_objects(comp)
    for rid in range(1, n + 1):
        sl = objs[rid - 1]
        # work in a padded window around the region, for speed on large images
        pad = max(cfg.node_buffer_px, cfg.road_buffer_px) + 2
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, comp.shape[0])
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, comp.shape[1])
        w = (slice(y0, y1), slice(x0, x1))
        region = comp[w] == rid
        area_px = int(region.sum())
        if area_px < cfg.min_region_px:
            continue

        # --- road association: how much of the change sits on a road --------
        ras = float((region & union_buf[w]).sum() / area_px)
        if ras < cfg.min_road_overlap:
            continue

        # --- added / removed / modified (slide 23 set logic, with tolerance) -
        D = _dilate(region, cfg.road_buffer_px)
        r1, r2 = roads_t1[w], roads_t2[w]
        added = int((D & ~r1 & r2).sum())       # C ∩ ¬R_T1 ∩ R_T2
        removed = int((D & r1 & ~r2).sum())     # C ∩ R_T1 ∩ ¬R_T2
        modified = int((D & r1 & r2).sum())     # C ∩ R_T1 ∩ R_T2
        counts = {"added": added, "removed": removed, "modified": modified}
        total = added + removed + modified
        if total == 0:
            ctype = "uncertain"
        elif added / total >= cfg.added_frac and added > removed:
            ctype = "added"
        elif removed / total >= cfg.removed_frac and removed > added:
            ctype = "removed"
        else:
            ctype = "modified"

        # --- magnitude: area, affected road length ---------------------------
        conf = float(change_prob[w][region].mean())
        area_m2 = area_px * cfg.gsd_m ** 2
        affected = skel[w] & D
        length_m = float(affected.sum() * cfg.gsd_m)
        L = min(1.0, length_m / cfg.length_ref_m)

        # --- connectivity impact K = N_affected / N_total (slide 25) -----------
        Dn = _dilate(region, cfg.node_buffer_px)
        affected_nodes = set(np.unique(node_lab[w][Dn & (node_lab[w] > 0)]).tolist())
        touched = set(np.unique(net_lab[w][D & skel[w]]).tolist()) - {0}
        total_nodes = set()
        for nl in touched:
            total_nodes |= nodes_of_net.get(nl, set())
        n_aff = len(affected_nodes & total_nodes)
        K = min(1.0, n_aff / cfg.nodes_max)          # beta_k / beta_max : junctions and end points inside the region
        A_norm = min(1.0, area_m2 / cfg.area_max_m2)  # A_k / A_max

        # --- RCRM:  S = wc*c + wa*A + wo*rho + wl*L + wn*I ---------------------------
        rrs = cfg.w_conf * conf + cfg.w_area * A_norm + cfg.w_assoc * ras + cfg.w_length * L + cfg.w_conn * K
        if rrs >= cfg.high_at:
            level = "High"
        elif rrs >= cfg.medium_at:
            level = "Medium"
        else:
            level = "Low"

        cy, cx = ndi.center_of_mass(region)
        out.append({
            "id": rid,
            "change_type": ctype,
            "area_px": area_px,
            "area_m2": round(area_m2, 1),
            "affected_road_length_m": round(length_m, 1),
            "confidence_C": round(conf, 3),
            "road_association_RAS": round(ras, 3),
            "area_norm_A": round(A_norm, 3),
            "length_norm_L": round(L, 3),
            "connectivity_K": round(K, 3),
            "affected_nodes": len(affected_nodes & total_nodes),
            "total_nodes": len(total_nodes),
            "priority_score": round(rrs, 3),
            "priority": level,
            "centroid_x": int(cx + x0),
            "centroid_y": int(cy + y0),
            "bbox": (int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)),
            "pixels_added": added, "pixels_removed": removed, "pixels_modified": modified,
        })
        kept[comp == rid] = rid

    out.sort(key=lambda r: r["priority_score"], reverse=True)
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return Analysis(regions=out, labels=kept, change_mask=change)
