import numpy as np
from PIL import Image

from slidegen.config import Settings
from slidegen.image_processing import denoise_image, enhance_image, prepare_for_analysis, process_image
from slidegen.models import ExtractedImage


def test_alpha_is_composited_on_white_not_black():
    rgba = Image.new("RGBA", (700, 700), (0, 0, 0, 0))            # fully transparent
    out = process_image(rgba, 1024)
    assert out.mode == "RGB" and out.getpixel((10, 10)) == (255, 255, 255)


def test_small_images_are_upsampled_and_large_downsampled():
    assert max(process_image(Image.new("RGB", (200, 100), "white"), 1024).size) >= 640
    assert max(process_image(Image.new("RGB", (3000, 1500), "white"), 1024).size) == 1024
    assert process_image(Image.new("RGB", (800, 600), "white"), 1024).size == (800, 600)


def test_denoise_reduces_noise_and_enhance_keeps_shape():
    rng = np.random.default_rng(0)
    clean = np.full((120, 120, 3), 128, np.uint8)
    noisy = np.clip(clean + rng.normal(0, 12, clean.shape), 0, 255).astype(np.uint8)
    assert denoise_image(noisy, 6).std() < noisy.std()
    assert denoise_image(noisy, 0) is noisy                         # strength 0 = disabled
    enhanced = enhance_image(clean)
    assert enhanced.shape == clean.shape and enhanced.dtype == np.uint8


def test_prepare_for_analysis_writes_processed_copy_and_keeps_original(make_image, tmp_path):
    path = make_image(size=(300, 200))
    img = ExtractedImage(path, 1, 1, 300, 200)
    before = open(path, "rb").read()
    prepare_for_analysis(img, tmp_path / "proc", Settings())
    assert img.processed_path.endswith(".png") and Image.open(img.processed_path).size[0] >= 640
    assert open(path, "rb").read() == before                        # original untouched (it goes on the slide)
