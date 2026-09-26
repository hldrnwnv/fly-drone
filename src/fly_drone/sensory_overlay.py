"""Render the exact saved neural vision outputs beside FPV flight footage."""

from __future__ import annotations

from math import pi

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _label(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font) -> None:
    x, y = xy
    box = draw.textbbox(xy, text, font=font)
    draw.rectangle((x - 7, y - 4, box[2] + 7, box[3] + 5), fill=(9, 17, 30))
    draw.text(xy, text, fill="white", font=font)


def sensory_screen(fpv: np.ndarray, chase: np.ndarray,
                   inverse_depth: np.ndarray, flow_xy: np.ndarray,
                   flow_valid: bool, observation: dict,
                   *, diagnostic_only: bool = False,
                   neural_input: bool = False,
                   retina_profile: np.ndarray | None = None) -> np.ndarray:
    """Return a 1280x720 2x2 display, with relative model outputs labeled."""
    height, width = fpv.shape[:2]
    font = _font(19)
    small = _font(16)
    canvas = Image.new("RGB", (2 * width, 2 * height), (9, 17, 30))
    canvas.paste(Image.fromarray(fpv), (0, 0))
    canvas.paste(Image.fromarray(chase), (width, 0))

    depth = np.asarray(inverse_depth, dtype=float)
    low, high = np.percentile(depth, [5, 95])
    scaled = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
    colors = np.stack((0.18 + 0.80 * scaled,
                       0.15 + 0.68 * scaled,
                       0.48 + 0.35 * (1 - scaled)), axis=-1)
    depth_image = Image.fromarray((255 * colors).astype(np.uint8))
    canvas.paste(depth_image.resize((width, height), Image.Resampling.BILINEAR),
                 (0, height))

    flow = np.asarray(flow_xy, dtype=float)
    magnitude = np.linalg.norm(flow, axis=-1)
    hue = ((np.arctan2(flow[..., 1], flow[..., 0]) + pi) / (2 * pi) * 255)
    value = np.clip(magnitude / 10, 0, 1) * 225 + 18 if flow_valid else np.zeros_like(hue)
    hsv = np.stack((hue, np.full_like(hue, 225), value), axis=-1).astype(np.uint8)
    flow_image = Image.fromarray(hsv, mode="HSV").convert("RGB")
    canvas.paste(flow_image.resize((width, height), Image.Resampling.BILINEAR),
                 (width, height))

    draw = ImageDraw.Draw(canvas)
    _label(draw, "FPV camera", (14, 12), font)
    if retina_profile is not None:
        profile = np.asarray(retina_profile, dtype=float)
        left, top, right, bottom = 8, height - 77, width - 8, height - 8
        draw.rectangle((left, top, right, bottom), fill=(9, 17, 30))
        draw.text((left + 9, top + 5), "Photoreceptor input: left to right",
                  fill="white", font=small)
        points = [(left + 9 + int(i * (right - left - 18) / max(len(profile) - 1, 1)),
                   bottom - 8 - int(np.clip(value, 0, 1) * 34))
                  for i, value in enumerate(profile)]
        if len(points) > 1:
            draw.line(points, fill=(104, 235, 170), width=2)
    _label(draw, "Fixed world view" if diagnostic_only or neural_input else "External view",
           (width + 14, 12), font)
    _label(draw, "Depth Anything V2: relative inverse depth", (14, height + 12), font)
    _label(draw, "RAFT Small: optical flow (color=direction)",
           (width + 14, height + 12), font)
    _label(draw, (f"t={observation.get('t', 0):.2f}s  "
                  f"decision={observation.get('decision_ms', 0):.0f}ms"),
           (width + 14, height + 44), small)
    if not flow_valid:
        _label(draw, "First frame: no flow yet", (width + 14, height + 70), small)

    if diagnostic_only or neural_input:
        cow_xy = observation.get("cow_xy")
        drone_xy = observation.get("drone_xy")
        if cow_xy and drone_xy:
            _label(draw, (f"cow ({cow_xy[0]:.1f}, {cow_xy[1]:.1f}) m   "
                          f"drone ({drone_xy[0]:.1f}, {drone_xy[1]:.1f}) m"),
                   (width + 14, 46), small)
        box_top = 2 * height - 82
        draw.rectangle((width, box_top, 2 * width, 2 * height), fill=(9, 17, 30))
        mode = (observation.get("visual_input") or {}).get("mode")
        description = ("FPV brightness -> photoreceptors; maps are display only"
                       if mode == "retina" else
                       "Flow -> T4/T5; depth display only" if mode == "flow" else
                       "Flow -> T4/T5; parallax -> LC15; looming -> LPLC2"
                       if neural_input else "Depth and flow: diagnostic display only")
        draw.text((width + 15, box_top + 10), description, fill="white", font=small)
        draw.text((width + 15, box_top + 40),
                  (f"tag bearing {observation.get('bearing', 0):+.2f} rad  "
                   f"turn {observation.get('steering_command', 0):+.2f}"),
                  fill="white", font=small)
    else:
        box_top = 2 * height - 122
        draw.rectangle((width, box_top, 2 * width, 2 * height), fill=(9, 17, 30))
        stimulus = observation.get("sensory_drive") or {}
        for row, axis in enumerate(("roll", "pitch", "yaw")):
            rate = float(stimulus.get(f"{axis}_rad_s", 0.0))
            y = box_top + 13 + 34 * row
            draw.text((width + 15, y), f"gyro {axis:5s} {rate:+5.2f} rad/s",
                      fill="white", font=small)
            center = width + 435
            draw.rectangle((center - 110, y + 6, center + 110, y + 18),
                           fill=(34, 50, 67))
            bar = int(np.clip(rate / 2, -1, 1) * 110)
            draw.rectangle((min(center, center + bar), y + 6,
                            max(center, center + bar), y + 18),
                           fill=(84, 225, 180) if bar >= 0 else (255, 170, 98))
    return np.asarray(canvas)


def odor_screen(fpv: np.ndarray, chase: np.ndarray, observation: dict) -> np.ndarray:
    """Show the two virtual antenna concentrations used at this decision."""
    height, width = fpv.shape[:2]
    canvas = Image.new("RGB", (2 * width, height))
    canvas.paste(Image.fromarray(fpv), (0, 0))
    canvas.paste(Image.fromarray(chase), (width, 0))
    draw = ImageDraw.Draw(canvas)
    small = _font(16)
    sample = observation["odor_sensor"]
    odor_input = observation.get("odor_input") or {}
    mode = odor_input.get("mode", "tag")
    left, top, right, bottom = width + 8, height - 106, 2 * width - 8, height - 8
    draw.rectangle((left, top, right, bottom), fill=(9, 17, 30))
    description = (("Tag visible: visual DNa02 steering" if odor_input.get("used") == "vision"
                    else "Tag masked: wind/odor fallback") if mode.startswith("fusion_") else
                   "Odor sampled; withheld from brain" if mode == "tag" else
                   "Odor -> ORN_DM1/VA2; no tag steering" if mode == "odor_only" else
                   "Wind only; no tag or odor steering" if mode == "wind_only" else
                   "Raw odor + wind; no tag steering" if mode == "raw_odor" else
                   "MaleCNS ORNs + wind; no tag steering" if mode.startswith("neural") else
                   "Odor -> ORN_DM1/VA2; tag steering present")
    draw.text((left + 10, top + 8), description, fill="white", font=small)
    for index, side in enumerate("LR"):
        y = top + 38 + 27 * index
        value = float(sample[side])
        draw.text((left + 10, y), f"{side}  {value:.2f}", fill="white", font=small)
        bar_left, bar_right = left + 120, right - 15
        draw.rectangle((bar_left, y + 5, bar_right, y + 17), fill=(37, 54, 66))
        draw.rectangle((bar_left, y + 5,
                        bar_left + int((bar_right - bar_left) * value), y + 17),
                       fill=(103, 226, 165))
    return np.asarray(canvas)
