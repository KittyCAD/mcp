import base64
import io
import tempfile
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp.utilities.types import Image
from mcp.types import ImageContent
from PIL import Image as PILImage

MAX_COLLAGE_IMAGES = 4
ImageFormat = Literal["jpeg", "png"]

_IMAGE_SUFFIXES: dict[ImageFormat, str] = {
    "jpeg": ".jpg",
    "png": ".png",
}


def create_image_collage(image_byte_list: list[bytes]) -> bytes:
    """Tile up to four equally sized images into a single JPEG.

    One image is passed through untouched, two are laid out side by side, and
    three or four fill a 2x2 grid. The finished grid is scaled back down so its
    width matches a single tile, keeping the result roughly the size of one
    view rather than growing with the number of views.
    """
    if not image_byte_list:
        raise ValueError("At least one image is required to create a collage.")
    if len(image_byte_list) > MAX_COLLAGE_IMAGES:
        raise ValueError(
            f"At most {MAX_COLLAGE_IMAGES} images can be tiled into a collage, "
            f"got {len(image_byte_list)}."
        )
    if len(image_byte_list) == 1:
        return image_byte_list[0]

    # Load images
    images = []
    for img_bytes in image_byte_list:
        img = PILImage.open(io.BytesIO(img_bytes))
        img = img.convert("RGB") if img.mode != "RGB" else img
        images.append(img)

    # Verify all are same size
    widths, heights = zip(*(img.size for img in images))
    if len(set(widths)) > 1 or len(set(heights)) > 1:
        for img in images:
            img.close()
        raise ValueError("All images must have the same dimensions.")

    img_w, img_h = images[0].size

    # Two images sit in a single row; three or four fill a 2x2 grid.
    columns = 2
    rows = (len(images) + columns - 1) // columns

    # An odd count leaves one cell empty. Filling it with the render's own
    # background colour keeps it from reading as a black panel in the collage.
    collage = PILImage.new(
        "RGB", (img_w * columns, img_h * rows), images[0].getpixel((0, 0))
    )
    for index, img in enumerate(images):
        collage.paste(img, ((index % columns) * img_w, (index // columns) * img_h))

    # Scale the grid back down so one tile's width is the collage's width.
    collage = collage.resize(
        (img_w, max(1, round(img_h * rows / columns))),
        PILImage.Resampling.LANCZOS,
    )

    # Save to bytes
    out = io.BytesIO()
    collage.save(out, format="JPEG", quality=95)
    collage_bytes = out.getvalue()

    # Cleanup
    for img in images:
        img.close()
    collage.close()
    out.close()

    return collage_bytes


def resize_image(img_bytes: bytes, max_dimension: int) -> bytes:
    """
    Resize an image so the longest side equals max_dimension, maintaining aspect ratio.

    Args:
        img_bytes: raw image bytes.
        max_dimension: The maximum width or height in pixels.

    Returns:
        Resized image bytes in JPEG format.
    """

    img = PILImage.open(io.BytesIO(img_bytes))
    img = img.convert("RGB") if img.mode != "RGB" else img

    w, h = img.size
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, PILImage.Resampling.LANCZOS)

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=95)

    result = out.getvalue()

    img.close()
    out.close()

    return result


def encode_image(img_bytes: bytes, image_format: ImageFormat = "jpeg") -> ImageContent:
    """
    Encode image bytes as MCP ImageContent with the correct media type.
    """
    img_obj = Image(data=img_bytes, format=image_format)
    return img_obj.to_image_content()


def save_image_bytes_to_disk(
    img_bytes: bytes,
    output_path: str | None = None,
    image_format: ImageFormat = "jpeg",
) -> str:
    """
    Write raw image bytes to disk.

    Args:
        img_bytes: The raw image bytes to write.
        output_path: The path where the image should be saved. If a directory is
            provided, a format-appropriate filename will be created there. If
            None, a temporary file will be created.
        image_format: The format of img_bytes. This controls the default suffix.

    Returns:
        str: The absolute path to the saved image file.
    """
    if output_path is None:
        # Create a temporary file
        _, temp_path = tempfile.mkstemp(suffix=_IMAGE_SUFFIXES[image_format])
        path = Path(temp_path)
    else:
        path = Path(output_path)

        # If path is a directory, create a default filename
        if path.is_dir():
            path = path / f"image{_IMAGE_SUFFIXES[image_format]}"

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

    # Write to disk
    path.write_bytes(img_bytes)

    return str(path.resolve())


def save_image_to_disk(image: ImageContent, output_path: str | None = None) -> str:
    """
    Saves an ImageContent object to disk.

    Args:
        image: The ImageContent object containing base64-encoded image data.
        output_path: The path where the image should be saved. If a directory is
            provided, a filename matching the image media type will be created
            there. If None, a temporary file will be created.

    Returns:
        str: The absolute path to the saved image file.
    """
    image_formats: dict[str, ImageFormat] = {
        "image/jpeg": "jpeg",
        "image/png": "png",
    }
    image_format = image_formats.get(image.mimeType)
    if image_format is None:
        raise ValueError(f"Unsupported image media type: {image.mimeType}")
    return save_image_bytes_to_disk(
        base64.b64decode(image.data), output_path, image_format
    )
