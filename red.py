import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en: {device}")

# --- 1. SINCRONIZACIÓN FÍSICA ESTRICTA ---
L_real = 0.00005
D_real = 1e-17
U_max = 10000.0

dx = L_real / 199.0 
dt_max = 1.0 / (4.0 * D_real * (1.0/(dx*dx)))
dt_real = dt_max * 0.9
T_max = 2000 * dt_real 

# --- 2. EL NUEVO UNIVERSO ADIMENSIONAL ---
D_norm = (D_real * T_max) / (L_real ** 2)
print(f"Coeficiente de difusión adimensional (D_norm): {D_norm:.5f}")

# --- 3. ARQUITECTURA DE LA RED BASE (SIN ANSATZ) ---
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
        # La red escupe la predicción en crudo, sin forzar la matemática inicial
        return self.red(torch.cat([x, y, t], dim=1))

modelo = PINN_Difusion().to(device)
optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

def calcular_derivadas(u, x, y, t):
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
    return u_t, u_xx, u_yy

# --- 4. ENTRENAMIENTO NORMALIZADO ---
def entrenar_pinn(epocas):
    for epoca in range(epocas):
        optimizador.zero_grad()

        # --- A. FÍSICA (Puntos generados entre 0.0 y 1.0) ---
        x_f = torch.rand(2000, 1, device=device, requires_grad=True)
        y_f = torch.rand(2000, 1, device=device, requires_grad=True)
        t_f = torch.rand(2000, 1, device=device, requires_grad=True)

        u_pred = modelo(x_f, y_f, t_f)
        u_t, u_xx, u_yy = calcular_derivadas(u_pred, x_f, y_f, t_f)
        
        residuo = u_t - D_norm * (u_xx + u_yy)
        loss_pde = torch.mean(residuo ** 2)

        # --- B. BORDES DIRICHLET (Perímetro completo congelado a 0.0) ---
        x_pared_v = torch.cat([torch.zeros(500, 1), torch.ones(500, 1)]).to(device)
        y_pared_v = torch.rand(1000, 1, device=device)
        t_pared_v = torch.rand(1000, 1, device=device)

        x_pared_h = torch.rand(1000, 1, device=device)
        y_pared_h = torch.cat([torch.zeros(500, 1), torch.ones(500, 1)]).to(device)
        t_pared_h = torch.rand(1000, 1, device=device)

        x_borde = torch.cat([x_pared_v, x_pared_h])
        y_borde = torch.cat([y_pared_v, y_pared_h])
        t_borde = torch.cat([t_pared_v, t_pared_h])

        loss_bc = torch.mean(modelo(x_borde, y_borde, t_borde) ** 2)

        # --- C. CONDICIÓN INICIAL (Evaluación estricta en t=0) ---
        x_ini = torch.rand(2000, 1, device=device)
        y_ini = torch.rand(2000, 1, device=device)
        t_ini = torch.zeros(2000, 1, device=device) # t=0
        
        u_ini_pred = modelo(x_ini, y_ini, t_ini)
        
        # OPCIÓN 1: Tablero de Ajedrez Térmico (Activa)
        # freq_x = 3.0 * torch.pi
        # freq_y = 3.0 * torch.pi
        # u_ini_real = (torch.sin(freq_x * x_ini) * torch.sin(freq_y * y_ini))**2

        # OPCIÓN 2: Dona Térmica (Descomenta estas líneas y comenta la Opción 1 si prefieres usarla)
        radio_medio = 0.25
        grosor = 0.08
        r = torch.sqrt((x_ini - 0.5)**2 + (y_ini - 0.5)**2)
        u_ini_real = torch.exp(-((r - radio_medio)**2) / (2.0 * grosor**2))

        loss_ic = torch.mean((u_ini_pred - u_ini_real) ** 2)

        # --- ENSAMBLAJE FINAL ---
        loss_total = loss_pde + loss_bc + loss_ic
        loss_total.backward()
        optimizador.step()

        if epoca % 500 == 0:
            print(f"Época {epoca} | Loss: {loss_total.item():.6f} | PDE: {loss_pde.item():.6f} | IC: {loss_ic.item():.6f} | BC: {loss_bc.item():.6f}")

entrenar_pinn(epocas=20000)

torch.save(modelo.state_dict(), 'pinn_difusion_normalizada.pth')
print("Entrenamiento completado y pesos guardados.")