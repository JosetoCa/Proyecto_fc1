#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#define M_PI 3.14159265358979323846
// #define CONDICION_AJEDREZ 
#define CONDICION_DONA

struct Particula { double x, y; bool activa; };

int main() {
    const double L = 0.00005, D = 1e-17;
    const double dt = 1.0 / (4.0 * D * (1.0/pow(L/199.0, 2))) * 0.9;
    const int num_particulas = 100000;
    
    std::random_device rd;
    std::mt19937 generador(rd());
    std::uniform_real_distribution<double> dist_espacio(0.0, L);
    std::uniform_real_distribution<double> dist_prob(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, std::sqrt(2.0 * D * dt));

    std::vector<Particula> enjambre(num_particulas);
    int i = 0;
    while(i < num_particulas) {
        double x_c = dist_espacio(generador), y_c = dist_espacio(generador);
        double prob = 0.0;
        
#ifdef CONDICION_AJEDREZ
        double freq = 3.0 * M_PI / L;
        prob = std::pow(std::sin(freq * x_c) * std::sin(freq * y_c), 2);
#elif defined(CONDICION_DONA)
        double r = std::sqrt(std::pow(x_c - L/2.0, 2) + std::pow(y_c - L/2.0, 2));
        prob = std::exp(-std::pow(r - 0.25*L, 2) / (2.0 * std::pow(0.08*L, 2)));
#endif
        if(dist_prob(generador) < prob) { enjambre[i++] = {x_c, y_c, true}; }
    }

    std::ofstream archivo("animacion_mc.csv");
    for (int t = 0; t < 2000; ++t) {
        for (int p = 0; p < num_particulas; ++p) {
            if (!enjambre[p].activa) continue;
            enjambre[p].x += gauss(generador);
            enjambre[p].y += gauss(generador);
            if (enjambre[p].x >= L || enjambre[p].x <= 0.0 || enjambre[p].y >= L || enjambre[p].y <= 0.0)
                enjambre[p].activa = false;
        }
        if (t % 20 == 0)
            for (auto& p : enjambre) if (p.activa) archivo << t << "," << p.x << "," << p.y << "\n";
    }
    return 0;
}