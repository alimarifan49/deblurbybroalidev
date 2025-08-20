# dl_backends/deblurganv2_backend.py
import torch
import cv2
import numpy as np
import os

def load_model(weights_dir=None, device='cpu'):
    """
    DeblurGANv2 biasanya punya generator dengan arsitektur tertentu (e.g. FPN-Inception).
    Kamu perlu:
      1) definisi arsitektur generator
      2) load_state_dict dari .pth di weights_dir
    Di sini kita sediakan placeholder.
    """
    if not weights_dir or not os.path.exists(weights_dir):
        raise RuntimeError("weights_dir tidak ditemukan.")
    raise RuntimeError("Implementasi DeblurGANv2 belum diisi. Tambahkan arsitektur + load weights.")

@torch.no_grad()
def infer(model, img_bgr, device='cpu'):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
    out = model(tensor)
    if isinstance(out, (list, tuple)):
        out = out[0]
    out = out.squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy()
    out_bgr = cv2.cvtColor((out*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return out_bgr
