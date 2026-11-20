import os
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

class CoastalControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Panel de Control - Monitorización y Pronóstico de Costa (CDIO1)")
        self.root.geometry("1100x750")

        # Configurar estilos visuales
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        self.create_header()
        self.create_tabs()
        self.create_console()

    def create_header(self):
        header_frame = ttk.Frame(self.root, padding=10)
        header_frame.pack(fill="x", padx=10, pady=10)
        
        title = ttk.Label(
            header_frame, 
            text="Panel de Control - Proyecto Línea de Costa Castelldefels", 
            font=("Helvetica", 16, "bold")
        )
        title.pack(side="left")
        
        project_label = ttk.Label(header_frame, text="Proyecto: costa_principal", font=("Helvetica", 10, "italic"))
        project_label.pack(side="right")

    def create_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)

        # 1. Pestaña Adquisición
        self.tab_descarga = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(self.tab_descarga, text="1. Adquisición CDSE")
        self.build_descarga_tab()

        # 2. Pestaña Procesamiento y Extracción
        self.tab_procesamiento = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(self.tab_procesamiento, text="2. Procesamiento y Extracción")
        self.build_procesamiento_tab()

        # 3. Pestaña Análisis y Modelos
        self.tab_analisis = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(self.tab_analisis, text="3. EDA y Modelos Paramétricos")
        self.build_analisis_tab()

        # 4. Pestaña LSTM & Pronóstico
        self.tab_lstm = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(self.tab_lstm, text="4. Pronóstico LSTM (2025)")
        self.build_lstm_tab()

    def build_descarga_tab(self):
        lbl_info = ttk.Label(
            self.tab_descarga, 
            text="Descarga de imágenes Sentinel-2 L2A desde Copernicus Data Space Ecosystem:",
            font=("Helvetica", 11, "bold")
        )
        lbl_info.pack(anchor="w", pady=5)

        form_frame = ttk.Frame(self.tab_descarga)
        form_frame.pack(fill="x", pady=10)

        ttk.Label(form_frame, text="Fecha Inicio (YYYY-MM-DD):").grid(row=0, column=0, sticky="w", pady=5)
        self.ent_inicio = ttk.Entry(form_frame, width=20)
        self.ent_inicio.insert(0, "2024-01-01")
        self.ent_inicio.grid(row=0, column=1, sticky="w", padx=10)

        ttk.Label(form_frame, text="Fecha Fin (YYYY-MM-DD):").grid(row=1, column=0, sticky="w", pady=5)
        self.ent_fin = ttk.Entry(form_frame, width=20)
        self.ent_fin.insert(0, "2024-12-31")
        self.ent_fin.grid(row=1, column=1, sticky="w", padx=10)

        ttk.Label(form_frame, text="Nubosidad Máxima Escena (%):").grid(row=2, column=0, sticky="w", pady=5)
        self.ent_nube = ttk.Entry(form_frame, width=20)
        self.ent_nube.insert(0, "40.0")
        self.ent_nube.grid(row=2, column=1, sticky="w", padx=10)

        btn_frame = ttk.Frame(self.tab_descarga)
        btn_frame.pack(fill="x", pady=15)

        btn_listar = ttk.Button(btn_frame, text="Buscar Productos (Solo Listar)", command=self.run_descarga_listar)
        btn_listar.pack(side="left", padx=5)

        btn_descargar = ttk.Button(btn_frame, text="Descargar Productos", command=self.run_descarga_real)
        btn_descargar.pack(side="left", padx=5)

    def build_procesamiento_tab(self):
        lbl_info = ttk.Label(
            self.tab_procesamiento, 
            text="Procesamiento de imágenes y extracción del borde costero:",
            font=("Helvetica", 11, "bold")
        )
        lbl_info.pack(anchor="w", pady=5)

        btn_s5 = ttk.Button(
            self.tab_procesamiento, 
            text="1. Procesar ZIPs (NDWI + Máscaras Binarias) [sesion5.py]", 
            command=lambda: self.run_command(["python", "sesion5.py"])
        )
        btn_s5.pack(anchor="w", fill="x", pady=5)

        btn_ext = ttk.Button(
            self.tab_procesamiento, 
            text="2. Extraer Bordes de Costa [extract_shorelines.py]", 
            command=lambda: self.run_command(["python", "extract_shorelines.py", "-p", "costa_principal"])
        )
        btn_ext.pack(anchor="w", fill="x", pady=5)

        btn_calc = ttk.Button(
            self.tab_procesamiento, 
            text="3. Calcular Distancias por Transecto [calculate_erosion.py]", 
            command=lambda: self.run_command(["python", "calculate_erosion.py", "-p", "costa_principal"])
        )
        btn_calc.pack(anchor="w", fill="x", pady=5)

        btn_vis = ttk.Button(
            self.tab_procesamiento, 
            text="4. Visualizar Distancias Extraídas [analyze.py]", 
            command=lambda: self.run_command(["python", "analyze.py", "-p", "costa_principal"])
        )
        btn_vis.pack(anchor="w", fill="x", pady=5)

    def build_analisis_tab(self):
        lbl_info = ttk.Label(
            self.tab_analisis, 
            text="Análisis exploratorio, limpieza de datos y ajuste de 3 modelos paramétricos:",
            font=("Helvetica", 11, "bold")
        )
        lbl_info.pack(anchor="w", pady=5)

        lbl_desc = ttk.Label(
            self.tab_analisis,
            text="Filtra transectos con NaNs, elimina outliers (IQR), calcula velocidad/aceleración\n"
                 "y compara 3 modelos: Lineal + FFT, Step (Gloria) + FFT, Sigmoide + FFT."
        )
        lbl_desc.pack(anchor="w", pady=5)

        btn_eda = ttk.Button(
            self.tab_analisis, 
            text="Ejecutar Análisis EDA y Ajustar Modelos Paramétricos [eda_and_parametric_models.py]", 
            command=lambda: self.run_command(["python", "eda_and_parametric_models.py"])
        )
        btn_eda.pack(anchor="w", fill="x", pady=15)

    def build_lstm_tab(self):
        lbl_info = ttk.Label(
            self.tab_lstm, 
            text="Predicción de erosión para Enero - Junio 2025 mediante Red LSTM:",
            font=("Helvetica", 11, "bold")
        )
        lbl_info.pack(anchor="w", pady=5)

        lbl_desc = ttk.Label(
            self.tab_lstm,
            text="Entrena una red Deep Learning LSTM mediante ventanas deslizantes\n"
                 "y genera la predicción a 6 meses comparándola con el modelo paramétrico y sus IC del 95%."
        )
        lbl_desc.pack(anchor="w", pady=5)

        btn_lstm = ttk.Button(
            self.tab_lstm, 
            text="Entrenar LSTM y Generar Pronóstico 2025 [forecasting_lstm.py]", 
            command=lambda: self.run_command(["python", "forecasting_lstm.py"])
        )
        btn_lstm.pack(anchor="w", fill="x", pady=15)

    def create_console(self):
        console_frame = ttk.LabelFrame(self.root, text="Consola de Salida / Registro", padding=10)
        console_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.console = scrolledtext.ScrolledText(console_frame, height=12, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9))
        self.console.pack(fill="both", expand=True)

    def log(self, text):
        self.console.insert(tk.END, text + "\n")
        self.console.see(tk.END)

    def run_descarga_listar(self):
        cmd = [
            "python", "descarga_copernicus.py",
            "--inicio", self.ent_inicio.get(),
            "--fin", self.ent_fin.get(),
            "--nubosidad", self.ent_nube.get(),
            "--solo-listar"
        ]
        self.run_command(cmd)

    def run_descarga_real(self):
        cmd = [
            "python", "descarga_copernicus.py",
            "--inicio", self.ent_inicio.get(),
            "--fin", self.ent_fin.get(),
            "--nubosidad", self.ent_nube.get()
        ]
        self.run_command(cmd)

    def run_command(self, cmd_list):
        def thread_target():
            self.log(f"\n[EJECUTANDO]: {' '.join(cmd_list)}")
            try:
                proc = subprocess.Popen(
                    cmd_list,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                for line in proc.stdout:
                    self.log(line.strip())
                proc.wait()
                if proc.returncode == 0:
                    self.log("[ÉXITO]: Proceso finalizado correctamente.")
                else:
                    self.log(f"[ERROR]: El proceso terminó con código {proc.returncode}.")
            except Exception as e:
                self.log(f"[EXCEPCIÓN]: {str(e)}")

        threading.Thread(target=thread_target, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = CoastalControlApp(root)
    root.mainloop()