import os
import re
from PIL import Image, ImageDraw, ImageFont
import easyocr

DATASET_CHOICE = "Fashion-MNIST"

RELEVANT_CONFIGS = [
    ("batchnorm", "1.0"),
    ("dataaug", "1.0"),
    ("dataaug", "3.0"),
    ("dropout", "0.3"),
    ("dropout", "0.6"),
    ("earlystopping", "5.0"),
    ("earlystopping", "12.0"),
    ("gaussiannoise", "0.05"),
    ("gaussiannoise", "0.2"),
    ("l1", "0.001"),
    ("l1", "0.01"),
    ("l2", "0.001"),
    ("l2", "0.01")
]

DISPLAY_NAMES = {
    "batchnorm": "BatchNorm",
    "dataaug": "DataAug",
    "dropout": "Dropout",
    "earlystopping": "EarlyStopping",
    "gaussiannoise": "GaussianNoise",
    "l1": "L1",
    "l2": "L2"
}

def es_config_relevante(metodo, valor):
    m_clean = metodo.lower().strip()
    v_clean = str(float(valor.strip()))
    for r_metodo, r_valor in RELEVANT_CONFIGS:
        try:
            r_v_clean = str(float(r_valor))
            if r_metodo == m_clean and r_v_clean == v_clean:
                return True
        except ValueError:
            if r_metodo == m_clean and r_valor == v_clean:
                return True
    return False

def procesar_repositorio_histogramas(carpeta_entrada, carpeta_salida, ruta_latex_output):
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        
    print("Inicializando EasyOCR con soporte GPU...")
    reader = easyocr.Reader(['en'], gpu=True)
    imagenes_procesadas = []
    
    archivos = [
        f for f in os.listdir(carpeta_entrada) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f != "temp_title_region.png"
    ]
    print(f"Se encontraron {len(archivos)} imágenes para analizar.")
    
    for archivo in archivos:
        ruta_img = os.path.join(carpeta_entrada, archivo)
        img = Image.open(ruta_img)
        w, h = img.size
        
        zona_titulo = img.crop((0, 0, w, 150))
        ruta_temporal = "temp_title_region.png"
        zona_titulo.save(ruta_temporal)
        
        resultados = reader.readtext(ruta_temporal, detail=0)
        texto_extraido = " ".join(resultados).lower().strip()
        
        if os.path.exists(ruta_temporal):
            os.remove(ruta_temporal)
            
        patron = r'(batchnorm|dataaug|dropout|earlystopping|gaussiannoise|l1|ll|l2)\s*=\s*([0-9.]+)'
        match = re.search(patron, texto_extraido)
        
        if not match:
            continue
            
        metodo_reg = match.group(1).replace("ll", "l1")
        valor_p = match.group(2)
        
        if not es_config_relevante(metodo_reg, valor_p):
            continue
            
        print(f" -> Procesando: {metodo_reg} = {valor_p} [{archivo}]")
        
        mitad_derecha = img.crop((1260, 0, 2520, 2160))
        lienzo = ImageDraw.Draw(mitad_derecha)
        
        lienzo.rectangle([0, 0, 1260, 150], fill="white")
        lienzo.rectangle([0, 190, 1260, 238], fill="white")
        lienzo.rectangle([0, 820, 1260, 868], fill="white")
        lienzo.rectangle([0, 1450, 1260, 1500], fill="white")
        
        try:
            fuente_titulo = ImageFont.truetype("arialbd.ttf", 44)
            fuente_sub = ImageFont.truetype("arialbd.ttf", 36)
        except IOError:
            try:
                fuente_titulo = ImageFont.truetype("arial.ttf", 44)
                fuente_sub = ImageFont.truetype("arial.ttf", 36)
            except IOError:
                fuente_titulo = ImageFont.load_default()
                fuente_sub = ImageFont.load_default()
                
        nombre_legible = DISPLAY_NAMES.get(metodo_reg, metodo_reg.upper())
        nuevo_titulo = f"{nombre_legible} = {valor_p} | Dataset: {DATASET_CHOICE}"
        
        bbox_t = fuente_titulo.getbbox(nuevo_titulo)
        w_t = bbox_t[2] - bbox_t[0]
        x_t = (1260 - w_t) // 2
        lienzo.text((x_t, 48), nuevo_titulo, fill="black", font=fuente_titulo)
        
        capas = [("conv1", 202), ("conv3", 832), ("fc1", 1462)]
        for nombre_capa, y_pos in capas:
            bbox_c = fuente_sub.getbbox(nombre_capa)
            w_c = bbox_c[2] - bbox_c[0]
            x_c = (1260 - w_c) // 2
            lienzo.text((x_c, y_pos), nombre_capa, fill="black", font=fuente_sub)
            
        reg_limpio = re.sub(r'[^a-zA-Z0-9]', '_', metodo_reg)
        dataset_limpio = re.sub(r'[^a-zA-Z0-9]', '_', DATASET_CHOICE.lower().strip())
        val_limpio = re.sub(r'[^a-zA-Z0-9]', '_', valor_p)
        
        nombre_nuevo_archivo = f"hist_{dataset_limpio}_{reg_limpio}_{val_limpio}.png"
        ruta_guardado = os.path.join(carpeta_salida, nombre_nuevo_archivo)
        mitad_derecha.save(ruta_guardado)
        
        imagenes_procesadas.append({
            "archivo": nombre_nuevo_archivo,
            "metodo": nombre_legible,
            "valor": valor_p,
            "label": f"fig:hist_{dataset_limpio}_{reg_limpio}_{val_limpio}"
        })
        
    bloques_latex = []
    print(f"\nGenerando código LaTeX secuencial para {len(imagenes_procesadas)} figuras...")
    
    for item in imagenes_procesadas:
        bloques_latex.append(
            f"\\begin{{figure}}[Visualización de la compresión de activaciones]"
            f"{{{item['label']}}}"
            f"{{Distribución \n    de activaciones por capa (\\texttt{{conv1}}, \\texttt{{conv3}} y \\texttt{{fc1}}), "
            f"sobre el conjunto de validación \n    de \\textit{{{DATASET_CHOICE}}}, con método de regularización {item['metodo']} y parámetro {item['valor']}.}}\n"
            f"    \\centering\n"
            f"    \\includegraphics[width=0.90\\textwidth]{{{carpeta_salida}/{item['archivo']}}}\n"
            f"\\end{{figure}}\n"
            f"\\FloatBarrier\n"
        )
        
    with open(ruta_latex_output, 'w', encoding='utf-8') as f:
        f.write("\n".join(bloques_latex))
    print(f"Proceso finalizado. Archivo LaTeX exportado en: {ruta_latex_output}")

if __name__ == "__main__":
    procesar_repositorio_histogramas(".", "proc", "figuras_output.tex")