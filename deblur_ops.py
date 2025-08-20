# deblur_ops.py
import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2
from skimage import img_as_float, img_as_ubyte, restoration

def _to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def _to_bgr(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def unsharp_mask(img_bgr, radius=2.0, amount=1.5):
    """
    radius: sigma Gaussian blur
    amount: seberapa kuat penajaman (1.0–2.0 umum)
    """
    img = img_as_float(_to_rgb(img_bgr))
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    sharp = np.clip(img + amount * (img - blurred), 0, 1)
    return _to_bgr(img_as_ubyte(sharp))

def motion_psf(length=15, angle=0, size=65):
    """
    Buat kernel motion blur sederhana.
    length: panjang garis
    angle: derajat (0 = horizontal)
    size: ukuran kernel (ganjil)
    """
    size = int(size) if int(size) % 2 == 1 else int(size) + 1
    psf = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    angle = np.deg2rad(angle)

    # buat garis
    x0 = center - int((length - 1) / 2 * np.cos(angle))
    y0 = center - int((length - 1) / 2 * np.sin(angle))
    x1 = center + int((length - 1) / 2 * np.cos(angle))
    y1 = center + int((length - 1) / 2 * np.sin(angle))
    cv2.line(psf, (x0, y0), (x1, y1), 1, 1)

    psf /= psf.sum() + 1e-8
    return psf

def wiener_deconvolution(img_bgr, length=15, angle=0, K=0.01):
    """
    Wiener deconvolution channel-wise.
    K: noise-to-signal power ratio (semakin besar → lebih halus/kurang artefak)
    """
    img = img_as_float(_to_rgb(img_bgr))
    psf = motion_psf(length=length, angle=angle, size=max(65, length*3 if length*3 % 2 else length*3+1))
    psf_f = fft2(psf, s=img.shape[:2])

    result = np.zeros_like(img)
    for c in range(3):
        channel = img[..., c]
        G = fft2(channel)
        H = psf_f
        H_conj = np.conj(H)
        F_hat = (H_conj / (H_conj * H + K)) * G
        f = np.real(ifft2(F_hat))
        # wrap-around ke rentang yang valid
        f = np.clip(f, 0, 1)
        result[..., c] = f

    return _to_bgr(img_as_ubyte(result))

# -------- Opsional: Hook ke backend DL --------
def deblur_with_mprnet(img_bgr, weights_path=None, device='cpu'):
    """
    Akan mencoba memanggil backend MPRNet jika tersedia.
    """
    try:
        from dl_backends.mprnet_backend import load_model, infer
    except Exception as e:
        raise RuntimeError(f"MPRNet backend belum siap: {e}")

    model = load_model(weights_path=weights_path, device=device)
    out = infer(model, img_bgr, device=device)
    return out

def deblur_with_deblurganv2(img_bgr, weights_dir=None, device='cpu'):
    """
    Akan mencoba memanggil backend DeblurGANv2 jika tersedia.
    """
    try:
        from dl_backends.deblurganv2_backend import load_model, infer
    except Exception as e:
        raise RuntimeError(f"DeblurGANv2 backend belum siap: {e}")

    model = load_model(weights_dir=weights_dir, device=device)
    out = infer(model, img_bgr, device=device)
    return out
