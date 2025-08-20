# dl_backends/mprnet_backend.py
import torch
import cv2
import numpy as np

def load_model(weights_path=None, device='cpu'):
    """
    Contoh: pakai torch.hub (jika repo menyediakan hubconf).
    Jika tidak, clone repo MPRNet dan import modulnya, lalu load_state_dict.
    """
    device = torch.device(device)
    try:
        # Misal repo sudah punya hub: (ganti owner/repo sesuai yang kamu gunakan)
        model = torch.hub.load('swz30/MPRNet', 'mprnet', pretrained=True)  # contoh; sesuaikan
        model.to(device).eval()
        return model
    except Exception:
        if not weights_path:
            raise RuntimeError("Tidak bisa load via torch.hub dan weights_path tidak diberikan.")
        # TODO: implementasi manual: inisiasi arsitektur MPRNet dan load_state_dict(weights_path)
        raise RuntimeError("Implementasi manual MPRNet belum diisi. Sediakan arsitektur + weights.")

@torch.no_grad()
def infer(model, img_bgr, device='cpu'):
    device = torch.device(device)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)

    out = model(tensor)  # sesuaikan jika model return tuple
    if isinstance(out, (list, tuple)):
        out = out[0]
    out = out.squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy()
    out_bgr = cv2.cvtColor((out*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return out_bgr
