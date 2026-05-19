import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

print("Cargando el censo de partículas. Esto puede tomar un momento...")
datos = pd.read_csv('animacion_mc.csv')

tiempos = datos['t'].unique()
total_frames = len(tiempos)

# 1. Configuración de la "Cuadrícula Analítica"
# Dividimos nuestra placa de 1x1 en 50x50 cajas para contar partículas
bins_x = 200
bins_y = 200
L = 0.00005

fig, ax = plt.subplots(figsize=(8, 7))
plt.subplots_adjust(bottom=0.25)

# 2. Extraer el primer fotograma para configurar colores iniciales
frame_inicial = datos[datos['t'] == tiempos[0]]

# numpy.histogram2d es la función mágica que cuenta las partículas por celda
matriz_conteo, bordes_x, bordes_y = np.histogram2d(
    frame_inicial['x'], frame_inicial['y'],
    bins=[bins_x, bins_y], 
    range=[[0, L], [0, L]] # LIMITAMOS LA VISTA AL TAMAÑO REAL
)
matriz_conteo = matriz_conteo.T

# Dibujamos el primer mapa de calor
# vmax limita el color máximo. Lo dividimos por 2 para que los colores sigan 
# siendo brillantes incluso cuando muchas partículas empiecen a fugarse.
cax = ax.imshow(matriz_conteo, cmap='inferno', origin='lower', extent=[0, L, 0, L], vmin=0, vmax=np.max(matriz_conteo)/60)
fig.colorbar(cax, label='Densidad de Partículas (Temperatura)')
titulo = ax.set_title(f'Paso temporal: {tiempos[0]} (Frame 0/{total_frames-1})')
ax.set_xlabel('Eje X (metros)')
ax.set_ylabel('Eje Y (metros)')

# 3. Crear la barra deslizante (Slider)
ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03])
slider_frame = Slider(
    ax=ax_slider,
    label='Frame',
    valmin=0,
    valmax=total_frames - 1,
    valinit=0,
    valstep=1
)

# 4. La función directora para actualizar cada frame
def actualizar_pantalla(indice_frame):
    idx = int(indice_frame)
    t_actual = tiempos[idx]
    
    # Extraer solo los datos de este tiempo
    frame = datos[datos['t'] == t_actual]
    
    # Volver a contar dónde están las partículas en este nuevo instante
    matriz_nueva, _, _ = np.histogram2d(
        frame['x'], frame['y'],
        bins=[bins_x, bins_y], 
        range=[[0, L], [0, L]]  # <--- CORREGIDO A 'L'
    )
    
    # Actualizar la imagen y el título
    cax.set_array(matriz_nueva.T)
    titulo.set_text(f'Paso temporal: {t_actual} (Frame {idx}/{total_frames-1})')
    fig.canvas.draw_idle()

slider_frame.on_changed(actualizar_pantalla)

# 5. Soporte para las flechas del teclado
def presionar_tecla(event):
    frame_actual = slider_frame.val
    if event.key == 'right' and frame_actual < total_frames - 1:
        slider_frame.set_val(frame_actual + 1)
    elif event.key == 'left' and frame_actual > 0:
        slider_frame.set_val(frame_actual - 1)

fig.canvas.mpl_connect('key_press_event', presionar_tecla)

print("¡Listo! Usa el slider o las flechas IZQ/DER para avanzar en el tiempo.")
plt.show()