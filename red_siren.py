import math
import torch
import torch.nn as nn

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en: {device}")

# ============================================================
# PARÁMETROS FÍSICOS
# ============================================================

L_real = 0.00005      # Longitud física de la placa (m)
D_real = 1e-17        # Coeficiente de difusión físico (m²/s)
U_max = 10000.0       # Escala máxima de temperatura

dx = L_real / 199.0

dt_max = 1.0 / (4.0 * D_real * (1.0 / (dx * dx)))
dt_real = 0.9 * dt_max

T_max = 2000 * dt_real

# ============================================================
# ADIMENSIONALIZACIÓN
# ============================================================

D_norm = (D_real * T_max) / (L_real**2)

print(f"D_norm = {D_norm:.6f}")

# ============================================================
# CAPA SIREN
# ============================================================

class SineLayer(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=30
    ):

        super().__init__()

        self.omega_0 = omega_0
        self.is_first = is_first

        self.linear = nn.Linear(
            in_features,
            out_features,
            bias=bias
        )

        self.init_weights()

    def init_weights(self):

        with torch.no_grad():

            if self.is_first:

                self.linear.weight.uniform_(
                    -1 / self.linear.in_features,
                     1 / self.linear.in_features
                )

            else:

                bound = (
                    math.sqrt(
                        6 / self.linear.in_features
                    )
                    / self.omega_0
                )

                self.linear.weight.uniform_(
                    -bound,
                     bound
                )

    def forward(self, x):

        return torch.sin(
            self.omega_0 * self.linear(x)
        )

# ============================================================
# RED PINN-SIREN
# ============================================================

class PINN_Difusion_SIREN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            SineLayer(
                3,
                128,
                is_first=True,
                omega_0=30
            ),

            SineLayer(
                128,
                128,
                omega_0=15
            ),

            SineLayer(
                128,
                128,
                omega_0=15
            ),

            SineLayer(
                128,
                128,
                omega_0=15
            ),

            nn.Linear(
                128,
                1
            )
        )

    def forward(self, x, y, t):

        entrada = torch.cat(
            [x, y, t],
            dim=1
        )

        return self.net(entrada)

# ============================================================
# CREACIÓN DEL MODELO
# ============================================================

modelo = PINN_Difusion_SIREN().to(device)

optimizador = torch.optim.Adam(
    modelo.parameters(),
    lr=1e-4
)

# ============================================================
# DERIVADAS AUTOMÁTICAS
# ============================================================

def calcular_derivadas(u, x, y, t):

    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    u_y = torch.autograd.grad(
        u,
        y,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True
    )[0]

    u_yy = torch.autograd.grad(
        u_y,
        y,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,
        retain_graph=True
    )[0]

    return u_t, u_xx, u_yy

# ============================================================
# ENTRENAMIENTO
# ============================================================

def entrenar_pinn(epocas=20000):

    for epoca in range(epocas):

        optimizador.zero_grad()

        # ----------------------------------------------------
        # 1. PUNTOS INTERNOS DEL DOMINIO
        # ----------------------------------------------------

        x_f = torch.rand(
            2000,
            1,
            device=device,
            requires_grad=True
        )

        y_f = torch.rand(
            2000,
            1,
            device=device,
            requires_grad=True
        )

        t_f = torch.rand(
            2000,
            1,
            device=device,
            requires_grad=True
        )

        u_f = modelo(
            x_f,
            y_f,
            t_f
        )

        u_t, u_xx, u_yy = calcular_derivadas(
            u_f,
            x_f,
            y_f,
            t_f
        )

        residuo = (
            u_t
            -
            D_norm * (u_xx + u_yy)
        )

        loss_pde = torch.mean(
            residuo**2
        )

        # ----------------------------------------------------
        # 2. CONDICIONES DE BORDE
        # ----------------------------------------------------

        x_pared_v = torch.cat([
            torch.zeros(500,1),
            torch.ones(500,1)
        ]).to(device)

        y_pared_v = torch.rand(
            1000,
            1,
            device=device
        )

        t_pared_v = torch.rand(
            1000,
            1,
            device=device
        )

        x_pared_h = torch.rand(
            1000,
            1,
            device=device
        )

        y_pared_h = torch.cat([
            torch.zeros(500,1),
            torch.ones(500,1)
        ]).to(device)

        t_pared_h = torch.rand(
            1000,
            1,
            device=device
        )

        x_borde = torch.cat([
            x_pared_v,
            x_pared_h
        ])

        y_borde = torch.cat([
            y_pared_v,
            y_pared_h
        ])

        t_borde = torch.cat([
            t_pared_v,
            t_pared_h
        ])

        u_borde = modelo(
            x_borde,
            y_borde,
            t_borde
        )

        loss_bc = torch.mean(
            u_borde**2
        )

        # ----------------------------------------------------
        # 3. CONDICIÓN INICIAL
        # ----------------------------------------------------

        x_ini = torch.rand(
            2000,
            1,
            device=device
        )

        y_ini = torch.rand(
            2000,
            1,
            device=device
        )

        t_ini = torch.zeros(
            2000,
            1,
            device=device
        )

        u_ini_pred = modelo(
            x_ini,
            y_ini,
            t_ini
        )

        radio_medio = 0.25
        grosor = 0.08

        r = torch.sqrt(
            (x_ini - 0.5)**2
            +
            (y_ini - 0.5)**2
        )

        u_ini_real = torch.exp(
            -((r - radio_medio)**2)
            /
            (2 * grosor**2)
        )

        loss_ic = torch.mean(
            (u_ini_pred - u_ini_real)**2
        )

        # ----------------------------------------------------
        # LOSS TOTAL
        # ----------------------------------------------------

        loss_total = (
            loss_pde
            +
            loss_bc
            +
            loss_ic
        )

        loss_total.backward()

        optimizador.step()

        if epoca % 500 == 0:

            print(
                f"Época {epoca:5d} | "
                f"Loss={loss_total.item():.6e} | "
                f"PDE={loss_pde.item():.6e} | "
                f"BC={loss_bc.item():.6e} | "
                f"IC={loss_ic.item():.6e}"
            )

# ============================================================
# ENTRENAMIENTO
# ============================================================

entrenar_pinn(20000)

# ============================================================
# GUARDAR PESOS
# ============================================================

torch.save(
    modelo.state_dict(),
    "pinn_siren_difusion.pth"
)

print("Entrenamiento finalizado.")