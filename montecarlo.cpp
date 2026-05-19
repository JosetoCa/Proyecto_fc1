#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>

// 1. Estructura de la partícula
struct Particula {
    double x;
    double y;
    bool activa; // Etiqueta de vida para condiciones de Dirichlet
};

int main() {
    // --- PARÁMETROS FÍSICOS (Sincronizados con Diferencias Finitas) ---
    const double L = 0.00005; 
    const double D = 1e-17;   
    const int Nx = 200; // Solo para calcular el mismo dt base

    // Cálculo estricto del tiempo
    const double dx = L / (Nx - 1);
    const double dt_max = 1.0 / (4.0 * D * (1.0/(dx*dx))); 
    const double dt = dt_max * 0.9; 

    const int pasos_temporales = 2000; 
    const int num_particulas = 100000; 

    // --- LA NUEVA FÍSICA: DESVIACIÓN ESTÁNDAR ---
    // En lugar de una distancia fija, calculamos el ancho de la campana de Gauss
    // que dicta qué tan lejos salta una partícula en promedio en un tiempo dt.
    double sigma = std::sqrt(2.0 * D * dt);

    // --- INICIALIZACIÓN ---
    std::vector<Particula> enjambre(num_particulas);
    for (int i = 0; i < num_particulas; ++i) {
        enjambre[i].x = L / 2.0;
        enjambre[i].y = L / 2.0;
        enjambre[i].activa = true; 
    }

    // --- MOTORES DE ESTADÍSTICA AVANZADA ---
    std::random_device rd; 
    std::mt19937 generador(rd());
    
    // Distribución Normal (Gaussiana) centrada en 0.0, con desviación 'sigma'
    std::normal_distribution<double> distribucion_gaussiana(0.0, sigma); 

    // --- CONFIGURACIÓN DE SALIDA ---
    std::ofstream archivo("animacion_mc.csv");
    archivo << "t,x,y\n";
    const int guardar_cada = 20; 

    std::cout << "Iniciando simulacion Monte Carlo Gaussiano...\n";

    // --- BUCLE PRINCIPAL DE LA SIMULACIÓN ---
    for (int t = 0; t < pasos_temporales; ++t) {
        
        for (int p = 0; p < num_particulas; ++p) {
            
            if (!enjambre[p].activa) continue; 
            
            // --- EL NUEVO SALTO CONTINUO ---
            // Extraemos dos números aleatorios de la campana de Gauss
            // Esto permite que la partícula patine libremente en diagonal, círculos, etc.
            enjambre[p].x += distribucion_gaussiana(generador);
            enjambre[p].y += distribucion_gaussiana(generador);

            // --- CONDICIÓN DE FRONTERA DE DIRICHLET ---
            if (enjambre[p].x >= L || enjambre[p].x <= 0.0 ||
                enjambre[p].y >= L || enjambre[p].y <= 0.0) {
                
                enjambre[p].activa = false; // Muere al tocar los bordes
            }
        }

        // Guardar estado
        if (t % guardar_cada == 0) {
            for (int p = 0; p < num_particulas; ++p) {
                if (enjambre[p].activa) {
                    archivo << t << "," << enjambre[p].x << "," << enjambre[p].y << "\n";
                }
            }
        }
    }

    archivo.close();
    std::cout << "Simulacion terminada. Datos exportados.\n";
    return 0;
}