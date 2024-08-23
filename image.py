from io import BytesIO
from sys import stderr
from typing import Any, TYPE_CHECKING

import numpy as np
from PIL import Image, ImageCms
from PIL.ImageCms import Intent

try:
    from numpy.typing import ArrayLike
except ImportError:
    if not TYPE_CHECKING:
        ArrayLike = Any

_SRGB = ImageCms.createProfile(colorSpace='sRGB')
_HRPC = ImageCms.FLAGS["HIGHRESPRECALC"]
_BPC  = ImageCms.FLAGS["BLACKPOINTCOMPENSATION"]

_INTENT_FLAGS = {
    Intent.PERCEPTUAL: _HRPC,
    Intent.RELATIVE_COLORIMETRIC: _HRPC | _BPC,
    Intent.ABSOLUTE_COLORIMETRIC: _HRPC
}

def _coalesce_intent(intent: Intent | int) -> Intent:
    if isinstance(intent, Intent):
        return intent

    match intent:
        case 0:
            return Intent.PERCEPTUAL
        case 1:
            return Intent.RELATIVE_COLORIMETRIC
        case 2:
            return Intent.SATURATION
        case 3:
            return Intent.ABSOLUTE_COLORIMETRIC
        case _:
            raise ValueError("invalid ImageCms intent")

def open_srgb(path: str) -> Image.Image:
    img = Image.open(path)

    match img.mode:
        case "RGBA" | "LA" | "PA" | "RGBa" | "La":
            mode = "RGBa"
        case _:
            if img.info.get("transparency") is None:
                mode = "RGB"
            else:
                mode = "RGBa"

    icc_raw = img.info.get("icc_profile")
    if icc_raw is not None:
        try:
            profile = ImageCms.ImageCmsProfile(BytesIO(icc_raw))

            if img.info.get("transparency") is not None:
                if img.mode in ("RGB", "RGBX", "BGR", "P") or img.mode.startswith("BGR;"):
                    img = img.convert("RGBA")
                elif img.mode in ("1", "L", "I") or img.mode.startswith("I;"):
                    img = img.convert("LA")
                else:
                    raise RuntimeError(f"transparency keying unexpected with mode {img.mode}")
            elif img.mode == "P":
                img = img.convert("RGB")
            elif img.mode in ("PA", "RGBa"):
                img = img.convert("RGBA")
            elif img.mode == "La":
                img = img.convert("LA")

            intent = Intent.RELATIVE_COLORIMETRIC
            if ImageCms.isIntentSupported(profile, intent, ImageCms.Direction.INPUT) != 1:
                intent = _coalesce_intent(ImageCms.getDefaultIntent(profile))

            flags = _INTENT_FLAGS.get(intent)
            if flags is not None:
                if img.mode == mode:
                    ImageCms.profileToProfile(
                        img,
                        profile,
                        _SRGB,
                        renderingIntent=intent,
                        inPlace=True,
                        flags=flags
                    )
                else:
                    img = ImageCms.profileToProfile(
                        img,
                        profile,
                        _SRGB,
                        renderingIntent=intent,
                        outputMode=mode,
                        flags=flags
                    )
            else:
                print(f"WARNING: \"{path}\": unsupported intent {intent}, assuming sRGB", file=stderr)
        except ImageCms.PyCMSError as ex:
            print(f"WARNING: \"{path}\": {ex}, assuming sRGB", file=stderr)

    if img.mode != mode:
        img = img.convert(mode)

    return img

def calculate_fit(
    in_width: int,
    in_height: int,
    out_width: int,
    out_height: int,
    max_scale: float | None = 1.0,
) -> tuple[int, int, float]:
    scale = min(out_width / in_width, out_height / in_height)
    if max_scale is not None:
        scale = min(scale, max_scale)

    return (
        min(round(in_width * scale), out_width),
        min(round(in_height * scale), out_height),
        scale
    )

def fit(
    image: Image.Image,
    width: int,
    height: int,
    *,
    max_scale: float | None = 1.0,
    resampling: Image.Resampling = Image.Resampling.LANCZOS,
    reducing_gap: float | None = None
) -> Image.Image:
    wnew, hnew, _ = calculate_fit(image.width, image.height, width, height, max_scale)
    if wnew == image.width and hnew == image.height:
        return image

    return image.resize((wnew, hnew), resample=resampling, reducing_gap=reducing_gap)

def calculate_center(
    in_width: int,
    in_height: int,
    out_width: int,
    out_height: int,
) -> tuple[int, int, int, int]:
    assert out_height >= in_height and out_width >= in_width

    top = (out_height - in_height) // 2
    bottom = top + in_height
    left = (out_width - in_width) // 2
    right = left + in_width

    return left, top, right, bottom

def calculate_crop(
    in_width: int,
    in_height: int,
    out_width: int,
    out_height: int,
) -> tuple[int, int, int, int]:
    return calculate_center(
        min(in_width, out_width),
        min(in_height, out_height),
        out_width,
        out_height
    )

def crop(
    image: Image.Image,
    width: int,
    height: int,
) -> Image.Image:
    return image.crop(calculate_crop(image.width, image.height, width, height))

def write_chw(image: ArrayLike | Image.Image, array: np.ndarray) -> None:
    np.copyto(array, np.asarray(image).transpose((2, 0, 1)), casting="safe")
    array /= 255.0

def load_image(path: str, xb_res: int, xq_res: int) -> tuple[np.ndarray, np.ndarray]:
    img = open_srgb(path)
    ch = 4 if img.mode == "RGBa" else 3

    imgr = fit(img, xb_res, xb_res, reducing_gap=3.0)
    imgc = crop(img, xq_res, xq_res)

    rl, rt, rr, rb = calculate_center(imgr.width, imgr.height, xb_res, xb_res)
    npr = np.zeros((ch, xb_res, xb_res), dtype=np.float16)
    write_chw(imgr, npr[:, rt:rb, rl:rr])
    npr = npr[:3, :, :]

    cl, ct, cr, cb = calculate_center(imgc.width, imgc.height, xq_res, xq_res)
    npc = np.zeros((ch, xq_res, xq_res), dtype=np.float16)
    write_chw(imgc, npc[:, ct:cb, cl:cr])
    npc = npc[:3, :, :]

    return npr, npc
