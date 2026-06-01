import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ============================================================
# 1. REPLICACIÓN DE LA ARQUITECTURA SIREN
# ============================================================
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = math.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class PINN_Difusion_SIREN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(3, 128, is_first=True, omega_0=30),
            SineLayer(128, 128, omega_0=15),
            SineLayer(128, 128, omega_0=15),
            SineLayer(128, 128, omega_0=15),
            nn.Linear(128, 1)
        )

    def forward(self, x, y, t):
        entrada = torch.cat([x, y, t], dim=1)
        return self.net(entrada)

# ============================================================
# 2. CÁLCULO DE ESCALAS FÍSICAS REPLICADO
# ============================================================
L_real = 0.00005
D_real = 1e-17
U_max = 10000.0

# Se debe replicar exactamente la lógica de tu entrenamiento
dx = L_real / 199.0
dt_max = 1.0 / (4.0 * D_real * (1.0 / (dx * dx)))
dt_real = 0.9 * dt_max
T_max = 2000 * dt_real 

# ============================================================
# 3. INICIALIZACIÓN DEL MODELO
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = PINN_Difusion_SIREN().to(device)

# Carga de pesos estricta
try:
    modelo.load_state_dict(torch.load("pinn_siren_difusion.pth", map_location=device, weights_only=True))
except FileNotFoundError:
    print("Error: El archivo 'pinn_siren_difusion.pth' no se encontró en el directorio actual.")
    exit(1)

modelo.eval()

# ============================================================
# 4. CONFIGURACIÓN DEL ESPACIO Y LA ANIMACIÓN
# ============================================================
Nx, Ny = 150, 150
num_frames = 100

x_vec = torch.linspace(0.0, 1.0, Nx)
y_vec = torch.linspace(0.0, 1.0, Ny)
X, Y = torch.meshgrid(x_vec, y_vec, indexing='ij')

x_plano = X.flatten().unsqueeze(1).to(device)
y_plano = Y.flatten().unsqueeze(1).to(device)

tiempos_norm = np.linspace(0.0, 1.0, num_frames)

fig, ax = plt.subplots(figsize=(7, 6))

# Estado inicial (t = 0)
t_plano_ini = torch.full_like(x_plano, tiempos_norm[0]).to(device)
with torch.no_grad():
    u_pred_ini = modelo(x_plano, y_plano, t_plano_ini)

matriz_inicial = (u_pred_ini * U_max).cpu().numpy().reshape((Nx, Ny))

# Renderizado del lienzo base
cax = ax.imshow(
    matriz_inicial.T, 
    cmap='inferno', 
    origin='lower', 
    extent=[0, L_real, 0, L_real], 
    vmin=0, 
    vmax=U_max
)
cbar = fig.colorbar(cax, label='Temperatura (PINN SIREN)')
ax.set_xlabel('Eje X (metros)')
ax.set_ylabel('Eje Y (metros)')

# ============================================================
# 5. BUCLE DE ACTUALIZACIÓN (FRAME POR FRAME)
# ============================================================
def actualizar(frame):
    t_norm_actual = tiempos_norm[frame]
    t_plano = torch.full_like(x_plano, t_norm_actual).to(device)
    
    with torch.no_grad():
        u_pred = modelo(x_plano, y_plano, t_plano)
    
    # Aplicación de U_max para mapear del dominio [0,1] al dominio físico
    matriz_nueva = (u_pred * U_max).cpu().numpy().reshape((Nx, Ny))
    cax.set_array(matriz_nueva.T)
    
    t_real = t_norm_actual * T_max
    ax.set_title(f'Tiempo físico: {t_real:.2e}s | Frame {frame}/{num_frames-1}')
    
    return [cax]

ani = animation.FuncAnimation(
    fig, 
    actualizar, 
    frames=num_frames, 
    interval=60,  # 60ms entre fotogramas (aprox. 16 FPS)
    blit=False, 
    repeat=True
)

plt.show()