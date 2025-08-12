import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk, ImageGrab
import torchvision.transforms as transforms
from model import UNET
from utils import load_checkpoint
from skimage.exposure import match_histograms  # Para el preprocesamiento

# Configuración global
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 560
IMAGE_WIDTH = 690

# --- Funciones compartidas ---

def overlay_mask(image, mask, alpha=0.5):
    """Superpone la máscara (canal verde) sobre la imagen original."""
    mask_colored = np.zeros_like(image, dtype=np.uint8)
    mask_colored[:, :, 1] = mask
    return cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)

def preprocess_frame(frame):
    """Preprocesa el frame para enviarlo al modelo."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ])
    return transform(frame_pil).unsqueeze(0).to(DEVICE)

# --- Funciones para preprocesamiento (histogram specification) ---

def find_ultrasound_cone(image):
    """
    Detecta la región de cono en una imagen de ultrasonido.
    Retorna una máscara binaria del cono.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No se encontraron contornos en la imagen.")
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=-1)
    return mask

def histogram_specification(source_image, target_image, source_mask, target_mask):
    """
    Ajusta el histograma del target_image para que se asemeje al source_image, 
    aplicándolo solo en las regiones definidas por las máscaras.
    """
    source_cone = cv2.bitwise_and(source_image, source_image, mask=source_mask)
    target_cone = cv2.bitwise_and(target_image, target_image, mask=target_mask)
    matched_cone = match_histograms(target_cone, source_cone, multichannel=True)
    result = np.zeros_like(target_image)
    for i in range(3):
        result[:, :, i] = cv2.bitwise_and(matched_cone[:, :, i], matched_cone[:, :, i], mask=target_mask)
    return result

# --- Clase principal de la aplicación ---

class SegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentación de Órganos en Ultrasonido")
        self.root.geometry("600x400")
        self.root.configure(bg="#e0f7fa")
        
        # Variables de configuración
        self.mode_var = tk.StringVar(value="camara")
        self.checkpoint_path = None
        self.video_path = None
        self.model = None
        
        # Variables para activar/desactivar procesamiento
        self.processing_active = tk.BooleanVar(value=True)
        self.preprocessing_active = tk.BooleanVar(value=False)
        self.preprocess_source_path = None
        self.preprocess_source_image = None
        self.preprocess_source_mask = None
        
        self.create_widgets()
    
    def create_widgets(self):
        titulo = tk.Label(self.root, text="Prueba de Segmentación de Ultrasonido", font=("Arial", 16, "bold"), bg="#e0f7fa")
        titulo.pack(pady=5)
        
        # Selección de modo de entrada
        modos_frame = tk.LabelFrame(self.root, text="Modo de Entrada", bg="#e0f7fa")
        modos_frame.pack(padx=10, pady=5, fill="x")
        tk.Radiobutton(modos_frame, text="Cámara en Vivo", variable=self.mode_var, value="camara", bg="#e0f7fa").pack(anchor="w")
        tk.Radiobutton(modos_frame, text="Sección de Pantalla", variable=self.mode_var, value="pantalla", bg="#e0f7fa").pack(anchor="w")
        tk.Radiobutton(modos_frame, text="Vídeo Almacenado", variable=self.mode_var, value="video", bg="#e0f7fa").pack(anchor="w")
        
        # Botones para checkpoint y video
        checkpoint_btn = ttk.Button(self.root, text="Seleccionar Checkpoint", command=self.select_checkpoint)
        checkpoint_btn.pack(pady=3)
        
        self.video_btn = ttk.Button(self.root, text="Seleccionar Vídeo", command=self.select_video, state="disabled")
        self.video_btn.pack(pady=3)
        self.mode_var.trace("w", self.mode_changed)
        
        # Checkbutton para activar/desactivar la segmentación
        seg_cb = tk.Checkbutton(self.root, text="Activar Segmentación", variable=self.processing_active, bg="#e0f7fa")
        seg_cb.pack(pady=2)
        
        # Checkbutton para activar/desactivar el preprocesamiento
        pre_cb = tk.Checkbutton(self.root, text="Activar Preprocesamiento", variable=self.preprocessing_active, bg="#e0f7fa")
        pre_cb.pack(pady=2)
        
        # Botón para seleccionar imagen fuente para preprocesamiento
        pre_source_btn = ttk.Button(self.root, text="Seleccionar imagen fuente (Preprocesamiento)", command=self.select_preprocess_source)
        pre_source_btn.pack(pady=3)
        
        # Botón para iniciar
        start_btn = ttk.Button(self.root, text="Iniciar", command=self.start_processing)
        start_btn.pack(pady=10)
        
        # Área de estado
        self.status_label = tk.Label(self.root, text="Esperando selección...", bg="#e0f7fa")
        self.status_label.pack(pady=5)
    
    def mode_changed(self, *args):
        mode = self.mode_var.get()
        if mode == "video":
            self.video_btn.config(state="normal")
        else:
            self.video_btn.config(state="disabled")
    
    def select_checkpoint(self):
        path = filedialog.askopenfilename(title="Seleccionar Checkpoint", filetypes=[("Archivos pth", "*.pth"), ("Todos los archivos", "*.*")])
        if path:
            self.checkpoint_path = path
            self.status_label.config(text=f"Checkpoint: {path}")
    
    def select_video(self):
        path = filedialog.askopenfilename(title="Seleccionar Vídeo", filetypes=[("Archivos de video", "*.mp4;*.avi"), ("Todos", "*.*")])
        if path:
            self.video_path = path
            self.status_label.config(text=f"Vídeo: {path}")
    
    def select_preprocess_source(self):
        path = filedialog.askopenfilename(title="Seleccionar imagen fuente", filetypes=[("Imagenes", "*.png;*.jpg;*.jpeg"), ("Todos", "*.*")])
        if path:
            self.preprocess_source_path = path
            self.preprocess_source_image = cv2.imread(path)
            try:
                self.preprocess_source_mask = find_ultrasound_cone(self.preprocess_source_image)
            except Exception as e:
                messagebox.showerror("Error", f"Error al procesar imagen fuente: {e}")
            self.status_label.config(text=f"Fuente preproc: {path}")
    
    def load_model(self):
        if not self.checkpoint_path:
            messagebox.showerror("Error", "Debes seleccionar un checkpoint.")
            return False
        self.status_label.config(text="Cargando modelo...")
        self.model = UNET(in_channels=3, out_channels=1).to(DEVICE)
        checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE)
        load_checkpoint(checkpoint, self.model)
        self.model.eval()
        self.status_label.config(text="Modelo cargado.")
        return True
    
    def start_processing(self):
        if not self.load_model():
            return
        
        mode = self.mode_var.get()
        if mode == "camara":
            source = 0
        elif mode == "video":
            if not self.video_path:
                messagebox.showerror("Error", "Debes seleccionar un vídeo.")
                return
            source = self.video_path
        else:  # modo "pantalla"
            source = None
        
        # Se ejecuta el procesamiento en un hilo separado
        threading.Thread(target=self.run, args=(mode, source), daemon=True).start()
    
    def run(self, mode, source):
        window_name = "Segmentación"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        if mode in ["camara", "video"]:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                messagebox.showerror("Error", "No se pudo abrir la fuente de vídeo.")
                return
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Lista de imágenes a mostrar (si se activa alguna opción se agregan)
                result_images = [frame.copy()]  # se muestra el frame original
                
                # Si la segmentación está activa, se procesa con el modelo
                if self.processing_active.get():
                    seg_frame = self.process_frame_model(frame)
                    result_images.append(seg_frame)
                
                # Si el preprocesamiento está activo y se ha seleccionado imagen fuente, se procesa
                if self.preprocessing_active.get() and self.preprocess_source_image is not None:
                    try:
                        target_mask = find_ultrasound_cone(frame)
                        pre_frame = histogram_specification(self.preprocess_source_image, frame, self.preprocess_source_mask, target_mask)
                    except Exception as e:
                        pre_frame = frame.copy()
                    result_images.append(pre_frame)
                
                combined = np.hstack(result_images)
                cv2.imshow(window_name, combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyWindow(window_name)
        elif mode == "pantalla":
            # Selección interactiva de la región con cv2.selectROI
            screenshot = ImageGrab.grab()
            screenshot_np = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            roi = cv2.selectROI("Selecciona la región de la pantalla y presiona ENTER", screenshot_np, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Selecciona la región de la pantalla y presiona ENTER")
            if roi[2] <= 0 or roi[3] <= 0:
                messagebox.showerror("Error", "No se seleccionó una región válida.")
                return
            x1, y1, w, h = roi
            x2, y2 = x1 + w, y1 + h
            
            while True:
                # Capturamos la región seleccionada
                img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                result_images = [frame.copy()]
                if self.processing_active.get():
                    seg_frame = self.process_frame_model(frame)
                    result_images.append(seg_frame)
                if self.preprocessing_active.get() and self.preprocess_source_image is not None:
                    try:
                        target_mask = find_ultrasound_cone(frame)
                        pre_frame = histogram_specification(self.preprocess_source_image, frame, self.preprocess_source_mask, target_mask)
                    except Exception as e:
                        pre_frame = frame.copy()
                    result_images.append(pre_frame)
                combined = np.hstack(result_images)
                cv2.imshow(window_name, combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyWindow(window_name)
    
    def process_frame_model(self, frame):
        """Procesa un frame aplicando el modelo de segmentación y superpone la máscara."""
        input_tensor = preprocess_frame(frame)
        with torch.no_grad():
            pred_mask = torch.sigmoid(self.model(input_tensor))
            pred_mask = (pred_mask > 0.5).float().cpu().squeeze().numpy()
        mask_resized = cv2.resize((pred_mask * 255).astype(np.uint8), (frame.shape[1], frame.shape[0]))
        overlayed = overlay_mask(frame, mask_resized)
        return overlayed

if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()
