import base64
import io
import tempfile
from pathlib import Path

from mcp.server.fastmcp.utilities.types import Image
from mcp.types import ImageContent
from PIL import Image as PILImage


def create_image_collage(image_byte_list: list[bytes]) -> bytes:
    assert len(image_byte_list) == 4, (
        "Exactly 4 images are required to create a 2x2 collage."
    )

    # Load images
    images = []
    for img_bytes in image_byte_list:
        img = PILImage.open(io.BytesIO(img_bytes))
        img = img.convert("RGB") if img.mode != "RGB" else img
        images.append(img)

    # Verify all are same size
    widths, heights = zip(*(img.size for img in images))
    if len(set(widths)) > 1 or len(set(heights)) > 1:
        raise ValueError("All images must have the same dimensions.")

    img_w, img_h = images[0].size

    # Create blank canvas 2x2
    collage = PILImage.new("RGB", (img_w * 2, img_h * 2))
    positions = [
        (0, 0),  # Top-left
        (img_w, 0),  # Top-right
        (0, img_h),  # Bottom-left
        (img_w, img_h),  # Bottom-right
    ]

    for img, pos in zip(images, positions):
        collage.paste(img, pos)

    # Scale down by 2x
    collage = collage.resize((img_w, img_h), PILImage.Resampling.LANCZOS)

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


def encode_image(img_bytes: bytes) -> ImageContent:
    """
    Encodes a PIL Image to a format compatible with ImageContent.
    """
    img_obj = Image(data=img_bytes, format="jpeg")
    return img_obj.to_image_content()


def save_image_bytes_to_disk(img_bytes: bytes, output_path: str | None = None) -> str:
    """
    Writes raw JPEG image bytes to disk.

    Args:
        img_bytes: The raw JPEG image bytes to write.
        output_path: The path where the image should be saved. If a directory is
            provided, a file named 'image.jpg' will be created in that directory.
            If None, a temporary file will be created.

    Returns:
        str: The absolute path to the saved image file.
    """
    if output_path is None:
        # Create a temporary file
        _, temp_path = tempfile.mkstemp(suffix=".jpg")
        path = Path(temp_path)
    else:
        path = Path(output_path)

        # If path is a directory, create a default filename
        if path.is_dir():
            path = path / "image.jpg"

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
            provided, a file named 'image.jpg' will be created in that directory.
            If None, a temporary file will be created.

    Returns:
        str: The absolute path to the saved image file.
    """
    return save_image_bytes_to_disk(base64.b64decode(image.data), output_path)
