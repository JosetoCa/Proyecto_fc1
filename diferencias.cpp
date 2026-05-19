#include <vector>
#include <iostream>
#include <utility>
#include <fstream>

class SimuladorDifusion2D {
private:
    int Nx, Ny;
    // Orden de declaración: dx, dy, dt, D
    double dx, dy, dt, D;
    std::vector<double> u_actual;
    std::vector<double> u_siguiente;

    inline int idx(int i, int j) const {
        return i + j * Nx;
    }

public:
    // Lista de inicialización corregida para coincidir con la declaración (dt antes que D)
    SimuladorDifusion2D(int nx, int ny, double lx, double ly, double coef_D, double delta_t) 
        : Nx(nx), Ny(ny), dt(delta_t), D(coef_D) {
        
        dx = lx / (Nx - 1);
        dy = ly / (Ny - 1);
        
        u_actual.resize(Nx * Ny, 0.0);
        u_siguiente.resize(Nx * Ny, 0.0);
    }

    void establecer_condicion_inicial(int i, int j, double valor) {
        if(i >= 0 && i < Nx && j >= 0 && j < Ny) {
            u_actual[idx(i, j)] = valor;
        }
    }

    void resolver_paso_temporal() {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                double d2u_dx2 = (u_actual[idx(i+1, j)] - 2.0 * u_actual[idx(i, j)] + u_actual[idx(i-1, j)]) / (dx * dx);
                double d2u_dy2 = (u_actual[idx(i, j+1)] - 2.0 * u_actual[idx(i, j)] + u_actual[idx(i, j-1)]) / (dy * dy);
                
                u_siguiente[idx(i, j)] = u_actual[idx(i, j)] + D * dt * (d2u_dx2 + d2u_dy2);
            }
        }
        u_actual.swap(u_siguiente); 
    }

    // Método para leer la concentración en un nodo específico
    double obtener_valor(int i, int j) const {
        if(i >= 0 && i < Nx && j >= 0 && j < Ny) return u_actual[idx(i, j)];
        return 0.0;
    }
};

int main() {
    const int Nx = 200;
    const int Ny = 200;
    const double Lx = 0.00005;
    const double Ly = 0.00005;
    const double D = 0.00000000000000001;
    
    const double dx = Lx / (Nx - 1);
    const double dy = Ly / (Ny - 1);
    const double dt_max = 1.0 / (2.0 * D * (1.0/(dx*dx) + 1.0/(dy*dy)));
    const double dt = dt_max * 0.9; 

    SimuladorDifusion2D sim(Nx, Ny, Lx, Ly, D, dt);

    int centro_x = Nx / 2;
    int centro_y = Ny / 2;
    sim.establecer_condicion_inicial(centro_x, centro_y, 10000.0);

    // --- CONFIGURACIÓN DE LA ANIMACIÓN ---
    std::ofstream archivo("animacion_difusion.csv");
    archivo << "t,x,y,temperatura\n"; // Nueva columna 't'

    const int pasos_temporales = 2000;  // Corremos más tiempo
    const int guardar_cada = 20;        // Tomamos foto cada 20 pasos

    std::cout << "Iniciando simulacion...\n";

    for (int t = 0; t < pasos_temporales; ++t) {
        
        // El motor de física calcula un paso más
        sim.resolver_paso_temporal();

        // Operador módulo: Si 't' es exactamente divisible por 20...
        if (t % guardar_cada == 0) {
            
            // ... tomamos la foto y la guardamos en el CSV
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    archivo << t << "," 
                        << i * dx << ","   // Multiplicamos por dx para que sea en metros
                        << j * dy << ","   // Multiplicamos por dy para que sea en metros
                        << sim.obtener_valor(i, j) << "\n";
                }
            }
        }
    }

    archivo.close();
    std::cout << "Simulacion terminada. Datos guardados.\n";
    return 0;
}