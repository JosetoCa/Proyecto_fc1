import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

print("Cargando datos. Esto puede tomar unos segundos...")
datos = pd.read_csv('animacion_difusion.csv')

# Descubrir las dimensiones
Nx = 200
Ny = 200
L = 0.00005
tiempos = datos['t'].unique()
total_frames = len(tiempos)

# 1. Configurar la figura dejando espacio en la parte inferior para los controles
fig, ax = plt.subplots(figsize=(8, 7))
plt.subplots_adjust(bottom=0.25) 

# Extraer el primer fotograma
frame_inicial = datos[datos['t'] == tiempos[0]]
matriz = frame_inicial['temperatura'].values.reshape((Ny, Nx))

# Dibujar la imagen inicial
cax = ax.imshow(matriz, cmap='inferno', origin='lower', vmin=0)
fig.colorbar(cax, label='Temperatura')
titulo = ax.set_title(f'Paso temporal: {tiempos[0]} (Frame 0/{total_frames-1})')

# 2. Crear el área para el Slider (Barra deslizante)
ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03])
slider_frame = Slider(
    ax=ax_slider,
    label='Frame',
    valmin=0,
    valmax=total_frames - 1,
    valinit=0,
    valstep=1 # Para que salte de 1 en 1 entero
)

# 3. Función maestra que actualiza la pantalla
def actualizar_pantalla(indice_frame):
    # Asegurarnos de que el índice sea un entero
    idx = int(indice_frame)
    t_actual = tiempos[idx]
    
    # Extraer y transformar los datos de ese tiempo
    frame = datos[datos['t'] == t_actual]
    matriz = frame['temperatura'].values.reshape((Ny, Nx))
    
    # Actualizar la imagen y el texto
    cax.set_array(matriz)
    titulo.set_text(f'Paso temporal: {t_actual} (Frame {idx}/{total_frames-1})')
    fig.canvas.draw_idle() # Pedirle a la ventana que se vuelva a dibujar

# Conectar el slider con la función maestra
slider_frame.on_changed(actualizar_pantalla)

# 4. Magia extra: Conectar las flechas del teclado
def presionar_tecla(event):
    frame_actual = slider_frame.val
    if event.key == 'right' and frame_actual < total_frames - 1:
        slider_frame.set_val(frame_actual + 1)
    elif event.key == 'left' and frame_actual > 0:
        slider_frame.set_val(frame_actual - 1)

fig.canvas.mpl_connect('key_press_event', presionar_tecla)

print("¡Listo! Usa el slider en la parte inferior o las flechas IZQ/DER del teclado.")
plt.show()