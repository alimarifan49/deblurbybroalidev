# app.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from deblur_ops import unsharp_mask, wiener_deconvolution, deblur_with_mprnet, deblur_with_deblurganv2

APP_TITLE = "DeblurApp By BroaliDEV n TEAM"

class DeblurApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1100x650")

        self.img_bgr = None
        self.proc_bgr = None
        self.current_path = None

        self.method_var = tk.StringVar(value="Unsharp Mask")
        self.k_length = tk.IntVar(value=15)
        self.k_angle = tk.IntVar(value=0)
        self.k_noise = tk.DoubleVar(value=0.01)
        self.unsharp_radius = tk.DoubleVar(value=2.0)
        self.unsharp_amount = tk.DoubleVar(value=1.5)

        self.device_var = tk.StringVar(value="cpu")
        self.mprnet_weights = tk.StringVar(value="")
        self.deblurgan_dir = tk.StringVar(value="")

        self._build_ui()
        self._build_menu()  # ✅ tambahkan menu bar

    def _build_ui(self):
        top = tk.Frame(self.root, padx=8, pady=6)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=4)
        methods = ["Unsharp Mask", "Wiener (Motion)", "Deep: MPRNet", "Deep: DeblurGANv2"]
        ttk.Combobox(top, textvariable=self.method_var, values=methods, width=24, state="readonly").pack(side=tk.LEFT, padx=8)

        tk.Button(top, text="Run", command=self.run).pack(side=tk.LEFT, padx=6)
        tk.Button(top, text="Save Result", command=self.save_image).pack(side=tk.LEFT, padx=6)

        # Params panel
        params = tk.LabelFrame(self.root, text="Parameters", padx=8, pady=6)
        params.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        # Unsharp
        tk.Label(params, text="Unsharp radius").pack(anchor="w")
        tk.Scale(params, variable=self.unsharp_radius, from_=0.5, to=5.0, resolution=0.1, orient="horizontal", length=200).pack()
        tk.Label(params, text="Unsharp amount").pack(anchor="w")
        tk.Scale(params, variable=self.unsharp_amount, from_=0.5, to=3.0, resolution=0.1, orient="horizontal", length=200).pack()

        # Wiener
        tk.Label(params, text="Motion length").pack(anchor="w")
        tk.Scale(params, variable=self.k_length, from_=5, to=45, resolution=1, orient="horizontal", length=200).pack()
        tk.Label(params, text="Motion angle (deg)").pack(anchor="w")
        tk.Scale(params, variable=self.k_angle, from_=-90, to=90, resolution=1, orient="horizontal", length=200).pack()
        tk.Label(params, text="Wiener K (noise)").pack(anchor="w")
        tk.Scale(params, variable=self.k_noise, from_=0.001, to=0.1, resolution=0.001, orient="horizontal", length=200).pack()

        # DL options
        dl_frame = tk.LabelFrame(params, text="Deep Learning", padx=6, pady=6)
        dl_frame.pack(fill=tk.X, pady=8)
        tk.Label(dl_frame, text="Device (cpu/cuda)").pack(anchor="w")
        ttk.Combobox(dl_frame, textvariable=self.device_var, values=["cpu","cuda"], width=10, state="readonly").pack(anchor="w")

        tk.Button(dl_frame, text="Set MPRNet weights (.pth)", command=self.choose_mprnet).pack(fill=tk.X, pady=2)
        tk.Label(dl_frame, textvariable=self.mprnet_weights, wraplength=180, fg="gray").pack(anchor="w")

        tk.Button(dl_frame, text="Set DeblurGANv2 dir", command=self.choose_deblurgan).pack(fill=tk.X, pady=2)
        tk.Label(dl_frame, textvariable=self.deblurgan_dir, wraplength=180, fg="gray").pack(anchor="w")

        # Image panels
        canvas = tk.Frame(self.root, padx=8, pady=8)
        canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.panel_src = tk.LabelFrame(canvas, text="Source", padx=4, pady=4)
        self.panel_src.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.lbl_src = tk.Label(self.panel_src)
        self.lbl_src.pack(fill=tk.BOTH, expand=True)

        self.panel_dst = tk.LabelFrame(canvas, text="Result", padx=4, pady=4)
        self.panel_dst.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.lbl_dst = tk.Label(self.panel_dst)
        self.lbl_dst.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status = tk.StringVar(value="Ready.")
        tk.Label(self.root, textvariable=self.status, anchor="w").pack(fill=tk.X, side=tk.BOTTOM)

    def _build_menu(self):
        menubar = tk.Menu(self.root)

        # Help Menu
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=helpmenu)

        self.root.config(menu=menubar)

    def show_about(self):
        messagebox.showinfo(
            "About",
            "Design by BroaliDEV, Nesha, Apri\nfrom UMP PWT"
        )

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Gagal membaca file gambar.")
            return
        self.img_bgr = img
        self.current_path = path
        self.proc_bgr = None
        self.status.set(f"Opened: {os.path.basename(path)}")
        self._show_image(self.lbl_src, self.img_bgr)
        self._show_image(self.lbl_dst, None)

    def _show_image(self, label_widget, img_bgr):
        if img_bgr is None:
            label_widget.config(image="", text="(empty)")
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        # fit to panel size
        panel_w = max(200, label_widget.winfo_width() or 500)
        panel_h = max(200, label_widget.winfo_height() or 500)
        scale = min(panel_w / w, panel_h / h, 1.0)
        new_size = (int(w*scale), int(h*scale))
        vis = cv2.resize(img_rgb, new_size, interpolation=cv2.INTER_AREA)
        im = Image.fromarray(vis)
        imgtk = ImageTk.PhotoImage(image=im)
        label_widget.imgtk = imgtk
        label_widget.configure(image=imgtk)

    def run(self):
        if self.img_bgr is None:
            messagebox.showwarning("No Image", "Buka gambar terlebih dahulu.")
            return

        method = self.method_var.get()
        try:
            if method == "Unsharp Mask":
                out = unsharp_mask(
                    self.img_bgr,
                    radius=float(self.unsharp_radius.get()),
                    amount=float(self.unsharp_amount.get())
                )
            elif method == "Wiener (Motion)":
                out = wiener_deconvolution(
                    self.img_bgr,
                    length=int(self.k_length.get()),
                    angle=int(self.k_angle.get()),
                    K=float(self.k_noise.get())
                )
            elif method == "Deep: MPRNet":
                out = deblur_with_mprnet(
                    self.img_bgr,
                    weights_path=self.mprnet_weights.get() or None,
                    device=self.device_var.get()
                )
            elif method == "Deep: DeblurGANv2":
                out = deblur_with_deblurganv2(
                    self.img_bgr,
                    weights_dir=self.deblurgan_dir.get() or None,
                    device=self.device_var.get()
                )
            else:
                raise ValueError("Metode tidak dikenal.")

            self.proc_bgr = out
            self._show_image(self.lbl_dst, self.proc_bgr)
            self.status.set(f"Done: {method}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.set("Error.")

    def save_image(self):
        if self.proc_bgr is None:
            messagebox.showinfo("No Result", "Belum ada hasil untuk disimpan.")
            return
        base = "result.png" if not self.current_path else f"deblur_{os.path.basename(self.current_path)}"
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=base,
                                            filetypes=[("PNG","*.png"),("JPEG","*.jpg;*.jpeg"),("BMP","*.bmp"),("TIFF","*.tif;*.tiff")])
        if not path:
            return
        ok = cv2.imwrite(path, self.proc_bgr)
        if ok:
            self.status.set(f"Saved: {path}")
        else:
            messagebox.showerror("Error", "Gagal menyimpan gambar.")

    def choose_mprnet(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Weights","*.pth;*.pt"),("All files","*.*")])
        if path:
            self.mprnet_weights.set(path)

    def choose_deblurgan(self):
        path = filedialog.askdirectory()
        if path:
            self.deblurgan_dir.set(path)

if __name__ == "__main__":
    root = tk.Tk()
    app = DeblurApp(root)
    root.mainloop()
