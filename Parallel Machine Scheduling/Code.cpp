#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <cmath>
#include <map>
#include <numeric>
#include <tuple>
#include <chrono>
//-------------------------------
// Estructuras básicas
//-------------------------------
struct Maquina {
    int id;
    int tiempo_ocupado = 0;


    // Necesario para comparar Estados en std::map
    bool operator<(const Maquina& other) const {
        // Comparamos por id y luego por tiempo
        return std::tie(id, tiempo_ocupado) < std::tie(other.id, other.tiempo_ocupado);
    }
    bool operator==(const Maquina& other) const {
        return id == other.id && tiempo_ocupado == other.tiempo_ocupado;
    }
};

struct Tarea {
    int id;
    int tiempo;

    // Necesario para comparar Estados en std::map
    bool operator<(const Tarea& other) const {
        return id < other.id;
    }
    bool operator==(const Tarea& other) const {
        return id == other.id;
    }
};

struct Asignacion {
    int tarea_id;
    int maquina_id;
    int posicion;
};

struct Estado {
    std::vector<Maquina> M;
    std::vector<Tarea> T;
    std::vector<Asignacion> Asignaciones;

    // Necesario para usar Estado como clave en std::map.
    // Comparamos los vectores M y T.
    bool operator<(const Estado& other) const {
        // Compara primero las tareas restantes
        if (T != other.T) return T < other.T;
        // Si son iguales, compara el estado de las máquinas
        return M < other.M;
    }
    bool operator==(const Estado& other) const {
        return T == other.T && M == other.M;
    }
};

//--------------------------------
// Función de asignación
//--------------------------------
Estado asignarTarea(const Estado& estado, int tarea_id, int maquina_id) {

    // 1. Creamos un nuevo estado como una COPIA del estado padre.
    //    Trabajaremos sobre 'nuevo', 'estado' permanece intacto.
    Estado nuevo = estado;
    // 2. Buscar la tarea que queremos asignar (por su ID) en la lista de pendientes 'T'.
    //    'it_tarea' será un un puntero a la tarea dentro del vector.
    auto it_tarea = std::find_if(nuevo.T.begin(), nuevo.T.end(),
                                   [&](const Tarea& t){ return t.id == tarea_id; });
    // 3. Comprobación de seguridad: Si no encontramos la tarea
    //    (es decir, si el iterador llega al final del vector sin encontrarla)
    //    devolvemos el estado 'nuevo' tal como está (una copia del original).
    if (it_tarea == nuevo.T.end()) return nuevo;

    // 4. Guardamos una COPIA de la tarea (con su 'id' y 'tiempo') antes de borrarla.
    //    Necesitamos su 'tiempo' para sumarlo a la máquina.
    Tarea tarea = *it_tarea;

    // 5. Buscar la máquina a la que queremos asignar la tarea (por su ID).
    //    'it_maquina' será un iterador a la máquina dentro del vector 'M'.
    auto it_maquina = std::find_if(nuevo.M.begin(), nuevo.M.end(),
                                     [&](const Maquina& m){ return m.id == maquina_id; });
    // 6. Comprobación de seguridad: Si no encontramos la máquina
    //    devolvemos el estado 'nuevo' (que es una copia del original).
    if (it_maquina == nuevo.M.end()) return nuevo;

    // 7. Obtenemos una REFERENCIA (&) a la máquina encontrada.
    Maquina& maquina = *it_maquina;



    // --- Aplicamos los 3 cambios que definen el nuevo estado ---
    // 9. Añadimos la asignación al historial 'Asignaciones' del nuevo estado.
    nuevo.Asignaciones.push_back({tarea.id, maquina.id});
    // 10. Eliminamos la tarea de la lista de pendientes 'T' del nuevo estado.
    //     Usamos el iterador que encontramos en el paso 2.
    nuevo.T.erase(it_tarea);
    // 11. Actualizamos el tiempo de la máquina en el nuevo estado.
    //     Como 'maquina' es una referencia (ver paso 7), esto modifica
    //     directamente la máquina que está DENTRO de 'nuevo.M'.
    maquina.tiempo_ocupado += tarea.tiempo;
    // 12. Devolvemos el estado 'nuevo' ya modificado.
    return nuevo;
}

//--------------------------------
// Coste g(n)
//--------------------------------
/*
  Calcula el coste g(n) (makespan actual) de un estado.
  Devuelve el tiempo_ocupado máximo de entre todas las máquinas.
 */
int calcularCoste(const Estado& estado) {
    int max_tiempo = 0;
    for (const auto& m : estado.M)
        // Actualiza el makespan si la máquina actual está más cargada
        max_tiempo = std::max(max_tiempo, m.tiempo_ocupado);
    return max_tiempo;
}

//--------------------------------
// Heurística h(n)
//--------------------------------
int calcularHeuristica2(const Estado& estado) {
    int M = estado.M.size();

    // 1. Calcula la suma total del tiempo de procesamiento de todas las tareas pendientes.
    // (Esto es: Σ T_restantes)
    int suma_t_restantes = 0;
    for (const auto& t : estado.T) suma_t_restantes += t.tiempo;

    // 2. Calcula el makespan actual (C_actual), que es el tiempo de la máquina más cargada.
    int tiempo_max = 0;
    for (const auto& m : estado.M) tiempo_max = std::max(tiempo_max, m.tiempo_ocupado);

    // 3. Calcula el "espacio libre total" en el sistema.
    // Para cada máquina, calcula cuánto tiempo le falta para alcanzar el makespan actual
    // y suma todas esas diferencias. (Esto es: Σ (C_actual - L_j))
    double suma_espacio_libre = 0.0;
    for (const auto& m : estado.M) suma_espacio_libre += (tiempo_max - m.tiempo_ocupado);

    // 4. Calcula el "espacio libre promedio" dividiendo el total por el número de máquinas.
    double espacio_libre_promedio = suma_espacio_libre / M;

    // 5. Calcula la heurística:
    // (Carga promedio restante por máquina) - (Espacio libre promedio disponible)
    double heuristica = (static_cast<double>(suma_t_restantes)/M) - espacio_libre_promedio;

    // 6. Si la carga promedio es menor que el espacio libre, estimamos 0 coste adicional.
    if (heuristica < 0) heuristica = 0;
    return static_cast<int>(std::round(heuristica));
    //return 0;
}
//--------------------------------
// Generar sucesores
//--------------------------------
std::vector<Estado> generarSucesores(const Estado& estado) {

    // 1. Se crea un vector vacío para almacenar los estados hijos que se generen.
    std::vector<Estado> sucesores;
    // 2. Itera sobre cada 'tarea' en la lista de tareas pendientes (estado.T)
    for (const auto& tarea : estado.T) {
        // 3. Itera sobre cada 'maquina' en la lista de máquinas (estado.M)
        for (const auto& maquina : estado.M) {

            // 4. Para CADA combinación de (tarea, maquina), crea un nuevo estado sucesor.
            //    Llama a 'asignarTarea' para obtener el resultado de esa asignación.
            //    Añade ese nuevo estado al vector de sucesores.
            sucesores.push_back(asignarTarea(estado, tarea.id, maquina.id));
        }
    }
    // 5. Devuelve la lista completa de todos los estados hijos generados.
    return sucesores;
}

//--------------------------------
// Comparador para la priority_queue
//--------------------------------
struct Nodo {
    Estado estado;
    int g_cost; // Costo real g(n)
    int f_cost; // Costo total f(n) = g(n) + h(n)

    // Operador para la min-priority_queue
    // Queremos que el nodo con MENOR f_cost tenga MAYOR prioridad
    bool operator>(const Nodo& other) const {
        return f_cost > other.f_cost;
    }
};

//--------------------------------
// Algoritmo de Búsqueda: A*
//--------------------------------
Estado A_estrella(const Estado& estado_inicial) {

    // --- LÍNEA 1: 'open' es una min-heap ---
    // Declara la 'cola' (Lista Abierta) como una cola de prioridad mínima (min-heap).
    // Gracias a std::greater<Nodo>, siempre pondrá en 'top()' el Nodo con el menor f_cost.
    std::priority_queue<Nodo, std::vector<Nodo>, std::greater<Nodo>> cola; // 'open'

    // --- Lista Cerrada ---
    // Almacena el *mejor g_cost* encontrado hasta ahora para un estado expandido
    // Es un 'map' para poder buscar eficientemente un 'Estado' (clave)
    // y obtener su 'g_cost' (valor). Esencial para evitar ciclos y transposiciones.
    std::map<Estado, int> closed_list;

    // Calculamos costes iniciales
    int g_inicial = calcularCoste(estado_inicial);
    int h_inicial = calcularHeuristica2(estado_inicial);
    // Añadimos el nodo inicial a la cola 'open' para arrancar la búsqueda.
    // Se empaqueta en un 'Nodo' con su estado, g_cost, y f_cost (g+h).
    cola.push({estado_inicial, g_inicial, g_inicial + h_inicial});

    // --- while |open| > 0 ---
    // El bucle principal de A*: se ejecuta mientras haya nodos por explorar.
    while (!cola.empty()) {

        // --- n <- pop(open) ---
        // Obtiene una copia del nodo con el MENOR f_cost (el mejor candidato).
        Nodo actual = cola.top();
        // Elimina ese nodo de la cola para procesarlo.
        cola.pop();

        // --- if n = t ---
        // Comprobación de meta: ¿Es un estado final?
        // Si el vector de tareas pendientes 'T' está vacío, hemos asignado todo.
        if (actual.estado.T.empty()) {
            return actual.estado; // ¡Solución!
        }

        // --- if n \not in closed \lor g_closed(n) > g(n) ---
        // Buscamos si el estado 'actual' ya ha sido visitado y está en la lista cerrada.
        auto it_closed = closed_list.find(actual.estado);

        // Comprobamos si:
        // 1. El estado NO está en la lista cerrada (it_closed == closed_list.end())
        // 2. O si SÍ está, pero hemos encontrado un camino MEJOR (menor g_cost)
        if (it_closed == closed_list.end() || actual.g_cost < it_closed->second) {

            // --- insert(n, closed) ---
            // (Re)insertamos el estado en la lista cerrada con su *mejor* coste 'g'
            closed_list[actual.estado] = actual.g_cost;

            // --- Generar y añadir hijos ---
            for (auto& sucesor : generarSucesores(actual.estado)) {
                int g_sucesor = calcularCoste(sucesor);
                int h_sucesor = calcularHeuristica2(sucesor);

                // Añadimos el sucesor a la cola 'open' para ser procesado
                // La cola de prioridad lo ordenará automáticamente según su f_cost.
                cola.push({sucesor, g_sucesor, g_sucesor + h_sucesor});
            }
        }

    }
    // Si el bucle 'while' termina (cola vacía) sin encontrar una solución,
    // significa que no hay un camino a la meta (en este problema, no debería
    // ocurrir si hay tareas). Devolvemos el estado inicial como señal de fallo.

    return estado_inicial; // (no se encontró solución)
}

//--------------------------------
// Programa principal
//--------------------------------
int main() {
    Estado estado;
    int N = 4 ; // EDITAR SI SE QUIERE CAMBIAR EL NUMERO DE MÁQUINAS
    for (int i = 1; i <= N; i++) {
        estado.M.push_back({i});
    }

    // EDITAR SI SE QUIERE CAMBIAR EL LISTADO DE TAREAS A ASIGNAR
    std::vector<int> tiempos = {25, 22, 19, 17, 12, 12, 11, 10, 10, 9, 9, 8, 8, 7, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1};
  //std::vector<int> tiempos = {25, 22, 19, 17, 12, 12, 11, 10, 10, 9, 9, 8, 8, 7, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 3, 2, 5, 4, 5}; // DESCOMENTAR SI SE QUIERE PROBAR ESTE EJEMPLO

    for (int i = 0; i < (int)tiempos.size(); ++i) {
        estado.T.push_back({i + 1, tiempos[i]});
    }

    // SE EJECUTA EL ALGORITMO DE BÚSQUEDA
    auto start = std::chrono::steady_clock::now();
    Estado solucion = A_estrella(estado);
    auto end = std::chrono::steady_clock::now();


    //----------------Presntación de los resultaods----------------
    std::cout << "Asignaciones finales:\n";
    for (const auto& a : solucion.Asignaciones)
        std::cout << "Tarea " << a.tarea_id << " -> Maquina " << a.maquina_id<< "\n";


    std::cout << "\nTiempo ocupado de maquinas:\n";
    for (const auto& m : solucion.M)
        std::cout << "Maquina " << m.id << ": " << m.tiempo_ocupado << "\n";

    std::cout << "Makespan final: " << calcularCoste(solucion) << "\n";

    std::map<int, std::vector<int>> tareas_por_maquina;
    std::map<int, std::vector<int>> tiempos_por_maquina;

    for (const auto& a : solucion.Asignaciones) {
        int tiempo = tiempos[a.tarea_id - 1];  // Obtener el tiempo correspondiente
        tareas_por_maquina[a.maquina_id].push_back(a.tarea_id);
        tiempos_por_maquina[a.maquina_id].push_back(tiempo);
    }

    std::cout << "\nAsignaciones por maquina:\n";
    for (const auto& m : solucion.M) {
        std::cout << "Maquina " << m.id << ":\n";

        // Mostrar IDs de tareas
        std::cout << "  Tareas (IDs): ";
        for (size_t i = 0; i < tareas_por_maquina[m.id].size(); ++i) {
            std::cout << tareas_por_maquina[m.id][i];
            if (i != tareas_por_maquina[m.id].size() - 1) std::cout << ", ";
        }

        std::cout << "\n  Tiempos:      ";
        for (size_t i = 0; i < tiempos_por_maquina[m.id].size(); ++i) {
            std::cout << tiempos_por_maquina[m.id][i];
            if (i != tiempos_por_maquina[m.id].size() - 1) std::cout << ", ";
        }

        std::cout << "\n";
    }


    double duration_s = std::chrono::duration<double>(end - start).count(); //
    std::cout << "Tiempo de busqueda (s):  " << duration_s << " s\n"; //
    return 0;
}
