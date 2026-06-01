import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
import torch.nn as nn
import numpy as np

# 1. Arquitectura exacta
class PINN_Difusion(nn.Module):
    def __init__(self):
        super(PINN_Difusion, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(3, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, y, t):
        return self.red(torch.cat([x, y, t], dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = PINN_Difusion().to(device)

# CORRECCIÓN 1: Nombre de archivo y pesos correctos
modelo.load_state_dict(torch.load('pinn_siren_difusion.pth', weights_only=True))
modelo.eval()

# Parámetros físicos reales para la proyección
Nx = 200
Ny = 200
L_real = 0.00005
U_max = 10000.0
T_max = 2840.0 # Tiempo real aproximado simulado previamente

# CORRECCIÓN 2: Entradas espaciales adimensionalizadas
x_vec = torch.linspace(0.0, 1.0, Nx)
y_vec = torch.linspace(0.0, 1.0, Ny)
X, Y = torch.meshgrid(x_vec, y_vec, indexing='ij')

x_plano = X.flatten().unsqueeze(1).to(device)
y_plano = Y.flatten().unsqueeze(1).to(device)

# CORRECCIÓN 3: Evaluación temporal adimensionalizada
num_frames = 50
tiempos_norm = np.linspace(0.0, 1.0, num_frames)

fig, ax = plt.subplots(figsize=(8, 7))
plt.subplots_adjust(bottom=0.25)

def predecir_fotograma(t_norm):
    t_plano = torch.full_like(x_plano, t_norm).to(device)
    
    with torch.no_grad():
        u_pred = modelo(x_plano, y_plano, t_plano)
    
    # CORRECCIÓN 4: Desnormalización física
    return (u_pred * U_max).cpu().numpy().reshape((Nx, Ny))

matriz_inicial = predecir_fotograma(tiempos_norm[0])

# El parámetro extent emplea las unidades físicas L_real
cax = ax.imshow(matriz_inicial.T, cmap='inferno', origin='lower', extent=[0, L_real, 0, L_real], vmin=0, vmax=U_max)
fig.colorbar(cax, label='Temperatura (PINN)')
titulo = ax.set_title(f'Paso temporal: 0.000s (Frame 0/{num_frames-1})')
ax.set_xlabel('Eje X (metros)')
ax.set_ylabel('Eje Y (metros)')

ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03])
slider_frame = Slider(ax=ax_slider, label='Frame', valmin=0, valmax=num_frames - 1, valinit=0, valstep=1)

def actualizar_pantalla(val):
    idx = int(slider_frame.val)
    t_norm_actual = tiempos_norm[idx]
    
    matriz_nueva = predecir_fotograma(t_norm_actual)
    cax.set_array(matriz_nueva.T)
    
    # CORRECCIÓN 5: Proyección a tiempo absoluto
    t_real = t_norm_actual * T_max
    titulo.set_text(f'Paso temporal: {t_real:.3f}s (Frame {idx}/{num_frames-1})')
    fig.canvas.draw_idle()

slider_frame.on_changed(actualizar_pantalla)

def presionar_tecla(event):
    frame_actual = slider_frame.val
    if event.key == 'right' and frame_actual < num_frames - 1:
        slider_frame.set_val(frame_actual + 1)
    elif event.key == 'left' and frame_actual > 0:
        slider_frame.set_val(frame_actual - 1)

fig.canvas.mpl_connect('key_press_event', presionar_tecla)

plt.show()