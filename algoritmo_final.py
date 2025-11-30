###############################################
# IMPORTS
###############################################
import random
import math
from collections import defaultdict
from distancias import DistanceManager

from deap import base, creator, tools, algorithms
import webbrowser
import pandas as pd
from statistics import mean, stdev

# NOVAS IMPORTAÇÕES PARA TSP EXATO (Dynamic Programming)
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming 

from sklearn_extra.cluster import KMedoids

# --- CONFIGURAÇÕES GLOBAIS ---
OPEN_BROWSER_ON_SUMMARY = False
ENABLE_DETAILED_SUMMARY = True  # VARIAVEL GLOBAL QUE HABILITA/DESABILITA SUMÁRIOS (TXT e PRINTS)

# VARIÁVEL GLOBAL PARA MODO DE EXECUÇÃO
# 1: RODAR TABELA (simulate_and_build_table_df)
# 2: RODAR UMA SIMULAÇÃO ALEATÓRIA (run_random_simulation)
# 3: ABRIR TODOS OS ENDEREÇOS NO GOOGLE MAPS (Abre URL longa)
EXECUTION_MODE = 2


###############################################
# ADDRESS MODEL
# (Não alterado)
###############################################

class Address:
# ... (restante da classe Address)
    """
    Address object that can contain either:
      - raw real-world address dict, or
      - coordinates for testing (dummy distance)
    The object is hashable.
    """

    def __init__(self, raw=None, coord=None):
        self.raw = raw
        self.coord = coord

        # Raw must be hashable: convert to tuple if dict.
        if isinstance(raw, dict):
            self._raw_tuple = tuple(sorted(raw.items()))
        else:
            self._raw_tuple = raw

    @staticmethod
    def from_coord(x, y):
        return Address(raw=None, coord=(x, y))

    @staticmethod
    def from_real_address(addr_dict):
        return Address(raw=addr_dict, coord=None)

    def __eq__(self, other):
        if not isinstance(other, Address):
            return False
        return self.coord == other.coord and self._raw_tuple == other._raw_tuple

    def __hash__(self,):
        return hash((self.coord, self._raw_tuple))

    def __repr__(self):
        if self.coord is not None:
            return f"Address(coord={self.coord})"
        return f"Address(raw={self.raw})"

# ... (restante das classes Patient, Distance Managers, Helpers)
class Patient:
    def __init__(self, id, address, appointment_time, duration, city):
        """
        appointment_time: em minutos desde 00:00 (ex: 9:00 -> 540)
        duration: duração da consulta em minutos
        city: nome da cidade (string)
        """
        self.id = id
        self.address = address
        self.appointment_time = appointment_time
        self.duration = duration
        self.city = city

    def __repr__(self):
        return f"Patient(id={self.id}, addr={self.address}, app={self.appointment_time}, dur={self.duration}, city={self.city})"


###############################################
# DISTANCE MANAGERS
# (Não alterado)
###############################################

class DummyDistanceManager:
    """
    Computes Euclidean distances between Address.coord
    Returns distances in meters (you decide scale of coords).
    """
    def distance(self, a: Address, b: Address):
        (x1, y1) = a.coord
        (x2, y2) = b.coord
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def distance_matrix(self, addresses):
        n = len(addresses)
        M = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                M[i][j] = self.distance(addresses[i], addresses[j])
        return M


class RealDistanceManager:
    """
    Wrapper que adapta o DistanceManager real (Google Maps + cache)
    para o modelo Address usado pelo otimizador.
    """

    def __init__(self, distance_manager: DistanceManager):
        """
        distance_manager: instância da classe DistanceManager real.
        """
        self.dm = distance_manager

    def distance(self, a: Address, b: Address):
        """
        Recebe Address objects do otimizador e usa o cálculo real.
        Deve retornar distância em metros.
        """
        if a.raw is None or b.raw is None:
            raise ValueError(
                "RealDistanceManager só funciona com Address contendo raw dict "
                "(use Address.from_real_address)."
            )

        return self.dm.get_distance_between_points(a.raw, b.raw)

    def distance_matrix(self, addresses):
        """
        Recebe array de Address e retorna matriz real de distâncias.
        """
        return self.dm.get_matrix_from_addresses(addresses)


###############################################
# HELPERS DE FORMATAÇÃO
###############################################

def format_address_for_display(addr: Address) -> str:
    if addr.coord is not None:
        return f"Coord{addr.coord}"

    if not addr.raw:
        return str(addr)

    address_text = addr.raw.get("address") or addr.raw.get("road") or addr.raw.get("logradouro") or "Endereço desconhecido"
    city_text = addr.raw.get("city") or addr.raw.get("cidade") or "Cidade desconhecida"

    uf = None
    for k in ("uf", "state", "estado"):
        if k in addr.raw and addr.raw[k]:
            uf = addr.raw[k]
            break

    if not uf and isinstance(city_text, str) and "," in city_text:
        parts = [p.strip() for p in city_text.split(",", 1)]
        city_text = parts[0]
        uf = parts[1] if len(parts) > 1 else None

    if uf:
        return f"{address_text} - {city_text}/{uf}"
    else:
        return f"{address_text} - {city_text}"


def minutes_to_Hh_Mm_string(total_minutes: float) -> str:
    """
    Converte minutos (pode ser float) para formato 'Hh Mm'.
    Suporta valores negativos para exibir diferenças (ex: '-1h 20m').
    """
    sign = ""
    if total_minutes < 0:
        sign = "-"
        total_minutes = abs(total_minutes)
    
    mins = int(round(total_minutes))
    h = mins // 60
    m = mins % 60
    return f"{sign}{h}h {m}m"


def minutes_to_HHMM(total_minutes: float) -> str:
    """
    Converte minutos desde meia-noite para 'HH:MM' (zero-padded).
    """
    mins = int(round(max(0, total_minutes)))
    h = mins // 60
    m = mins % 60
    return f"{h:02d}:{m:02d}"


def meters_to_km_string(total_meters: float) -> str:
    """
    Converte metros para formato 'X.X km'.
    Suporta valores negativos para exibir diferenças.
    """
    sign = ""
    if total_meters < 0:
        sign = "-"
        total_meters = abs(total_meters)

    km = total_meters / 1000.0
    return f"{sign}{km:.1f} km"


###############################################
# ROUTING (Exact TSP)
# (Inalterado da modificação anterior)
###############################################

def exact_tsp_route(dm, start_addr: Address, patient_addrs):
    """
    Calcula a rota de custo mínimo (ciclo TSP) usando Dynamic Programming.
    Retorna a permutação de endereços de pacientes.
    A complexidade O(N^2 * 2^N) limita o uso para N <= ~20.
    """
    if not patient_addrs:
        return []

    # 1. Lista completa de endereços para o ciclo: [Ponto Inicial, P1, P2, ..., PN]
    all_addrs = [start_addr] + patient_addrs
    
    # 2. Matriz de distâncias
    distance_matrix = dm.distance_matrix(all_addrs)
    
    # O solver espera um array numpy
    D = np.array(distance_matrix)

    # 3. Resolve o TSP (o ciclo deve começar no índice 0 - start_addr)
    # Retorna a permutação de índices [0, i1, i2, ..., iN]
    permutation, _ = solve_tsp_dynamic_programming(D)
    
    # 4. Extrai a rota de endereços de pacientes (ignora o primeiro índice 0)
    # A permutação é a sequência de índices que representa a ordem de visita.
    route_addrs = []
    
    # Ignora o índice 0 (start_addr)
    for index in permutation[1:]:
        route_addrs.append(all_addrs[index])

    # A rota de endereços dos pacientes na ordem ideal é devolvida
    return route_addrs


###############################################
# GLOBAL SUMMARY STORAGE
# (Não alterado)
###############################################

SIM_SUMMARY = {
# ... (conteúdo omitido)
    "baseline": {
        "vans": [],
        "total_cost": None,
        "total_distance": None
    },
    "ga": {
        "vans": [],
        "total_cost": None,
        "assignment": None,
        "total_distance": None
    }
}


def clear_sim_summary():
    SIM_SUMMARY["baseline"] = {"vans": [], "total_cost": None, "total_distance": None}
    SIM_SUMMARY["ga"] = {"vans": [], "total_cost": None, "assignment": None, "total_distance": None}


###############################################
# FITNESS CALCULATION
# (Inalterado da modificação anterior)
###############################################

VELOCIDADE_INTRA = 25000
VELOCIDADE_INTER = 70000


def compute_travel_time_meters(dist_m: float, city_from: str, city_to: str) -> float:
# ... (conteúdo omitido)
    if city_from == city_to:
        return dist_m / VELOCIDADE_INTRA
    else:
        return dist_m / VELOCIDADE_INTER


def compute_van_cost(dm, start_address: Address, patients, mode=None, van_id=None):
# ... (conteúdo omitido)
    """
    Calcula custo (tempo total de espera/viagem dos passageiros) e distância.
    Retorna custo (tempo) total em minutos, para o fitness.
    A ordem da rota é determinada por um solver TSP exato.
    """
    if len(patients) == 0:
        if mode is not None:
            SIM_SUMMARY[mode]["vans"].append({
                "van_id": van_id, "route_display": [], "arrival_times_display": {},
                "patient_times_out_display": {}, "van_total_time_minutes": 0.0,
                "patients_info": [], "full_route_addresses": [start_address, start_address],
                "total_distance_meters": 0.0
            })
        return 0.0

    addr_to_patient = {p.address: p for p in patients}
    addrs = [p.address for p in patients]
    
    # MUDANÇA: Usa a rota TSP exata
    route = exact_tsp_route(dm, start_address, addrs) # Rota otimizada!

    current_addr = start_address
    cumulative_minutes = 0.0
    arrival_minutes_since_dep = {}
    total_route_distance_meters_ida = 0.0 

    start_city = None
    if start_address.raw and isinstance(start_address.raw, dict):
        start_city = start_address.raw.get("city") or start_address.raw.get("cidade") or None
    if not start_city and patients:
        start_city = patients[0].city

    for next_addr in route:
        next_patient = addr_to_patient[next_addr]
        dist_m = dm.distance(current_addr, next_addr)
        
        if current_addr == start_address:
            from_city = start_city
        else:
            cur_p = addr_to_patient.get(current_addr)
            from_city = cur_p.city if cur_p else start_city

        t_hours = compute_travel_time_meters(dist_m, from_city, next_patient.city)
        t_minutes = t_hours * 60.0
        cumulative_minutes += t_minutes
        arrival_minutes_since_dep[next_addr] = cumulative_minutes
        total_route_distance_meters_ida += dist_m
        current_addr = next_addr

    total_route_distance_meters = total_route_distance_meters_ida * 2.0
    total_time_to_last_minutes = cumulative_minutes

    required_departures = []
    for p in patients:
        delivery_time = arrival_minutes_since_dep[p.address]
        required = p.appointment_time - delivery_time
        required_departures.append(required)

    departure_minutes = max(0.0, min(required_departures))

    arrival_wallclock_minutes = {addr: departure_minutes + arrival_minutes_since_dep[addr] for addr in arrival_minutes_since_dep}

    finish_times = [p.appointment_time + p.duration for p in patients]
    inicio_retorno = max(finish_times)

    return_travel_minutes = total_time_to_last_minutes
    chegada_origem_minutes = inicio_retorno + return_travel_minutes
    time_out_per_passenger_minutes = max(0.0, chegada_origem_minutes - departure_minutes)

    total_cost_minutes = time_out_per_passenger_minutes * len(patients)

    if mode is not None:
        # Preenche SIM_SUMMARY (total_distance_meters é a distância ABSOLUTA da VAN)
        route_display = []
        route_display.append({"step": 0, "address_display": format_address_for_display(start_address), "time_display": minutes_to_HHMM(departure_minutes)})
        for idx, addr in enumerate(route, start=1):
            arr_wc = arrival_wallclock_minutes.get(addr, departure_minutes)
            route_display.append({"step": idx, "address_display": format_address_for_display(addr), "time_display": minutes_to_HHMM(arr_wc)})
        route_display.append({"step": len(route) + 1, "address_display": format_address_for_display(start_address), "time_display": minutes_to_HHMM(chegada_origem_minutes)})
        full_route_addresses = [start_address] + route + [start_address]
        patients_info = []
        for p in patients:
            arrival_wc = arrival_wallclock_minutes.get(p.address, departure_minutes)
            patients_info.append({
                "id": p.id, "address_display": format_address_for_display(p.address), "appointment_display": minutes_to_HHMM(p.appointment_time),
                "duration_display": minutes_to_Hh_Mm_string(p.duration), "arrival_display": minutes_to_HHMM(arrival_wc),
                "time_out_display": minutes_to_Hh_Mm_string(time_out_per_passenger_minutes)
            })
        
        SIM_SUMMARY[mode]["vans"].append({
            "van_id": van_id, "route_display": route_display,
            "arrival_times_display": {format_address_for_display(k): minutes_to_HHMM(v) for k, v in arrival_wallclock_minutes.items()},
            "patient_times_out_display": {format_address_for_display(k): minutes_to_Hh_Mm_string(time_out_per_passenger_minutes) for k in arrival_wallclock_minutes},
            "van_total_time_minutes": time_out_per_passenger_minutes, "patients_info": patients_info,
            "departure_minutes": departure_minutes, "chegada_origem_minutes": chegada_origem_minutes,
            "full_route_addresses": full_route_addresses, "total_distance_meters": total_route_distance_meters
        })

    return total_cost_minutes


def evaluate(individual, patients, num_vans, dm, start_address): # Calcula Média de Distância por Van
# ... (conteúdo omitido)
    """
    individual[i] = van id for patient i
    Returns tuple (cost_minutes,)
    """
    vans = [[] for _ in range(num_vans)]

    for i, van_id in enumerate(individual):
        vans[van_id].append(patients[i])

    cost = 0.0
    for v in vans:
        cost += compute_van_cost(dm, start_address, v)

    if hasattr(evaluate, "log_mode") and evaluate.log_mode:
        van_id = 0
        for v in vans:
            compute_van_cost(dm, start_address, v, mode="ga", van_id=van_id)
            van_id += 1
        SIM_SUMMARY["ga"]["total_cost"] = cost # Custo ABSOLUTO

        # MÉDIA da distância (por van)
        num_vans_used = len(SIM_SUMMARY["ga"]["vans"])
        total_distance_sum = sum(v["total_distance_meters"] for v in SIM_SUMMARY["ga"]["vans"])
        SIM_SUMMARY["ga"]["total_distance"] = total_distance_sum / num_vans_used if num_vans_used > 0 else 0.0

    return (cost,)


def k_medoids(distance_matrix, k, max_iter=100):
    """
    Implementação de K-Medoids usando scikit-learn-extra com matriz de distância pré-calculada.
    Retorna os índices dos medoides e os rótulos de cluster.
    """
    n = len(distance_matrix)
    if n == 0 or k <= 0:
        return [], []
    k = min(k, n)

    # 1. Converte a matriz de lista de listas para array numpy (exigido pelo sklearn)
    D = np.array(distance_matrix)

    # 2. Instancia KMedoids com a métrica 'precomputed'
    # O random_state é usado para reprodutibilidade.
    kmedoids = KMedoids(n_clusters=k, metric='precomputed', max_iter=max_iter, random_state=42)
    
    # 3. Faz o fit (treinamento/clusterização)
    # Passamos a matriz D como o 'X', e kmedoids sabe que ela já é a matriz de distância
    kmedoids.fit(D)
    
    # 4. Extrai os resultados
    # medoid_indices_ são os índices dos pontos que se tornaram medoides
    medoids = kmedoids.medoid_indices_.tolist() 
    # labels_ são os rótulos de cluster para cada ponto
    labels = kmedoids.labels_.tolist()

    return medoids, labels


###############################################
# BASELINE: CLUSTER BY K-MEDOIDS
# (Chama a nova função k_medoids)
###############################################

def baseline_by_city(patients, num_vans, dm, start_address): # Calcula Média de Distância por Van
# ... (conteúdo inalterado)
    clear_sim_summary()
    n = len(patients)
    if n == 0:
        van_id = 0
        while van_id < num_vans:
            compute_van_cost(dm, start_address, [], mode="baseline", van_id=van_id)
            van_id += 1
        SIM_SUMMARY["baseline"]["total_cost"] = 0.0
        SIM_SUMMARY["baseline"]["total_distance"] = 0.0
        return 0.0

    k = min(num_vans, n)
    addresses = [p.address for p in patients]
    D = dm.distance_matrix(addresses)
    medoids, labels = k_medoids(D, k)

    clusters = [[] for _ in range(k)]
    for idx, lab in enumerate(labels):
        clusters[lab].append(patients[idx])

    total_cost = 0.0
    SIM_SUMMARY["baseline"]["vans"] = []

    van_id = 0
    for cluster in clusters:
        if van_id >= num_vans: break
        total_cost += compute_van_cost(dm, start_address, cluster, mode="baseline", van_id=van_id)
        van_id += 1

    while van_id < num_vans:
        compute_van_cost(dm, start_address, [], mode="baseline", van_id=van_id)
        van_id += 1

    SIM_SUMMARY["baseline"]["total_cost"] = total_cost # Custo ABSOLUTO
    
    # MÉDIA da distância (por van)
    num_vans_used = len(SIM_SUMMARY["baseline"]["vans"])
    total_distance_sum = sum(v["total_distance_meters"] for v in SIM_SUMMARY["baseline"]["vans"])
    SIM_SUMMARY["baseline"]["total_distance"] = total_distance_sum / num_vans_used if num_vans_used > 0 else 0.0

    return total_cost


###############################################
# GENETIC ALGORITHM USING DEAP
# [Omitido por brevidade, inalterado]
###############################################

def solve_with_ga(patients, num_vans, dm, start_address, ngen=50):
# ... (conteúdo inalterado)
    clear_sim_summary()
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_van", lambda: random.randint(0, num_vans-1))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_van, n=len(patients))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, patients=patients, num_vans=num_vans, dm=dm, start_address=start_address)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=num_vans-1, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=80)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.1, ngen=ngen, halloffame=hof, verbose=False)

    best = hof[0]
    best_cost = evaluate(best, patients, num_vans, dm, start_address)[0]

    evaluate.log_mode = True
    _ = evaluate(best, patients, num_vans, dm, start_address)
    SIM_SUMMARY["ga"]["assignment"] = best

    return best, best_cost


###############################################
# SUMARIZAÇÃO
###############################################

def open_google_maps_route(route_addresses: list[Address]):
# ... (conteúdo inalterado)
    if not route_addresses or len(route_addresses) < 2: return None
    origem_address = route_addresses[0]
    destino_address = route_addresses[-1]
    paradas_addresses = route_addresses[1:-1]
    def get_address_string(addr: Address) -> str:
        if addr.raw and isinstance(addr.raw, dict):
            parts = [addr.raw.get("address") or addr.raw.get("road") or addr.raw.get("logradouro")]
            city = addr.raw.get("city") or addr.raw.get("cidade")
            uf = addr.raw.get("uf") or addr.raw.get("state") or addr.raw.get("estado")
            if city: parts.append(city)
            if uf: parts.append(uf)
            return ", ".join(filter(None, parts))
        elif addr.coord is not None: return f"{addr.coord[0]},{addr.coord[1]}"
        return format_address_for_display(addr)
    addresses_to_format = [origem_address] + paradas_addresses + [destino_address]
    partes_da_url = []
    for addr in addresses_to_format:
        partes_da_url.append(get_address_string(addr).replace(" ", "+").replace(",", ""))
    return f"https://www.google.com/maps/dir/{'/'.join(partes_da_url)}"


def summarize_results(mode="baseline", output="stdout"):
# ... (conteúdo inalterado)
    if mode not in SIM_SUMMARY: return

    summary = SIM_SUMMARY[mode]
    text_lines = []
    text_lines.append(f"===== SUMMARY [{mode.upper()}] =====\n")
    total_cost = summary.get("total_cost")
    if total_cost is not None:
        text_lines.append(f"Total cost (sum of passenger times out): {minutes_to_Hh_Mm_string(total_cost)}\n")

    total_distance = summary.get("total_distance")
    if total_distance is not None:
        text_lines.append(f"Average distance per van: {meters_to_km_string(total_distance)}\n")

    if mode == "ga" and summary.get("assignment") is not None:
        text_lines.append(f"Assignment (patient_index -> van): {summary['assignment']}\n")

    text_lines.append("\n--- VANS DETAILS ---\n")
    for van in summary["vans"]:
        text_lines.append(f"\nVan {van['van_id']}:\n")
        route_addresses = van.get("full_route_addresses")
        if route_addresses:
            gmaps_url = open_google_maps_route(route_addresses)
            if gmaps_url:
                text_lines.append(f"  [Google Maps] Rota no navegador:\n")
                text_lines.append(f"  {gmaps_url}\n")
                try:
                    if van["van_id"] == 0 and len(route_addresses) > 2 and output == "stdout" and OPEN_BROWSER_ON_SUMMARY:
                         webbrowser.open(gmaps_url)
                         text_lines.append(f"  (Rota da Van {van['van_id']} aberta no navegador.)\n")
                    else:
                        text_lines.append(f"  (Copie e cole a URL acima no seu navegador.)\n")
                except Exception:
                    text_lines.append(f"  (webbrowser falhou ao abrir. Copie e cole a URL.)\n")

        text_lines.append("  Route (IDA e VOLTA):\n")
        for step in van["route_display"]:
            step_num = step["step"]
            addr_disp = step["address_display"]
            time_disp = step["time_display"]
            if step_num == 0: text_lines.append(f"    Saída: {addr_disp}  (partida: {time_disp})\n")
            elif step_num == len(van["route_display"]) - 1: text_lines.append(f"    Retorno: {addr_disp}  (previsto chegada: {time_disp})\n")
            else:
                text_lines.append(f"    {step_num}) {addr_disp}\n")
                text_lines.append(f"       Chegada prevista: {time_disp}\n")

        text_lines.append("  Pacientes:\n")
        for pinfo in van["patients_info"]:
            text_lines.append(f"    - ID {pinfo['id']}: {pinfo['address_display']}\n")
            text_lines.append(f"       Horário consulta: {pinfo['appointment_display']}\n")
            text_lines.append(f"       Duração consulta: {pinfo['duration_display']}\n")
            text_lines.append(f"       Horário chegada van: {pinfo['arrival_display']}\n")
            text_lines.append(f"       Tempo fora (van): {pinfo['time_out_display']}\n")

        dep = van.get("departure_minutes")
        arr_origin = van.get("chegada_origem_minutes")
        if dep is not None and arr_origin is not None:
            text_lines.append(f"  Saída origem: {minutes_to_HHMM(dep)}\n")
            text_lines.append(f"  Chegada origem prevista: {minutes_to_HHMM(arr_origin)}\n")

        total_time_min = van.get("van_total_time_minutes", 0.0)
        text_lines.append(f"  Van total time (minutos por passageiro): {int(round(total_time_min))} min\n")

        van_dist = van.get("total_distance_meters", 0.0)
        text_lines.append(f"  Van total distance: {meters_to_km_string(van_dist)}\n")

    final_text = "".join(text_lines)

    if output == "stdout":
        print(final_text)
    else:
        with open(output, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"Sumário salvo em {output}")


def run_random_simulation(
    dm,
    loader,
    num_patients=10,
    num_vans=3,
    start_address=None,
    save_summaries=None
):
# ... (conteúdo inalterado)
    if save_summaries is None: save_summaries = ENABLE_DETAILED_SUMMARY

    if start_address is None:
        start_address = Address.from_real_address({
            'address': 'Praça Barão do Rio Branco, 12 - Pilar, Ouro Preto - MG',
            'city': 'Ouro Preto', 'uf': 'MG'
        })

    patients = []
    for pid in range(num_patients):
        real_addr = loader.get_random_address()
        appointment_minute = random.randint(8*60, 17*60)
        duration = random.randint(15, 40)
        patient = Patient(
            id=pid, address=Address.from_real_address(real_addr), appointment_time=appointment_minute,
            duration=duration, city=real_addr.get("city", "Desconhecida")
        )
        patients.append(patient)

    if save_summaries: print("\n=== Baseline (cluster by city) ===")
    baseline_cost = baseline_by_city(patients, num_vans, dm, start_address)
    baseline_distance = SIM_SUMMARY["baseline"]["total_distance"]
    if save_summaries: print("Baseline cost:", minutes_to_Hh_Mm_string(baseline_cost))

    if save_summaries:
        summarize_results(mode="baseline")
        summarize_results(mode="baseline", output="baseline_summary.txt")

    if save_summaries: print("\n=== Genetic Algorithm ===")
    best, best_cost = solve_with_ga(patients, num_vans, dm, start_address)
    ga_distance = SIM_SUMMARY["ga"]["total_distance"]
    if save_summaries: print("Best cost:", minutes_to_Hh_Mm_string(best_cost))

    if save_summaries:
        summarize_results(mode="ga")
        summarize_results(mode="ga", output="ga_summary.txt")

    return {
        "num_patients": len(patients), # Passa o número real de pacientes
        "num_vans": num_vans,
        "start_address": start_address,
        "baseline_cost": baseline_cost,
        "ga_cost": best_cost,
        "baseline_distance": baseline_distance,
        "ga_distance": ga_distance,
        "ga_best_solution": best
    }


def simulate_and_build_table_df(
    patient_range,
    van_range, # Alterado: aceita range de vans
    n_runs,
    dm,
    loader,
    start_address,
    simulation_fn
):
# ... (conteúdo inalterado)
    """
    Tempo: Média por Passageiro (TotalCost / num_patients)
    Distância: Média por Van
    """
    results = []

    for p in patient_range:
        for v in van_range: # Novo laço para iteração sobre o número de vans
            for _ in range(n_runs):
                sim = simulation_fn(
                    num_patients=p,
                    num_vans=v, # Passa o número de vans atual (v)
                    dm=dm,
                    loader=loader,
                    start_address=start_address,
                    save_summaries=ENABLE_DETAILED_SUMMARY 
                )
                results.append(sim)

    grouped = {}
    for r in results:
        key = (r["num_patients"], r["num_vans"])
        grouped.setdefault(key, [])
        grouped[key].append(r)

    def safe_stdev(v):
        return stdev(v) if len(v) > 1 else 0.0

    rows = []

    for (num_pat, vans), sims in sorted(grouped.items()):
        # Dados de Custo (Tempo) - CÁLCULO MÉDIO POR PASSAGEIRO
        # Custo por paciente em minutos: (TotalCost / num_patients)
        # O número de pacientes é pego do dicionário de resultado da simulação
        base_cost_vals_avg_pax = [s["baseline_cost"] / s["num_patients"] for s in sims if s["num_patients"] > 0]
        ga_cost_vals_avg_pax = [s["ga_cost"] / s["num_patients"] for s in sims if s["num_patients"] > 0]
        diff_cost_vals_avg_pax = [g - b for g, b in zip(ga_cost_vals_avg_pax, base_cost_vals_avg_pax)]

        # Dados de Distância (MÉDIA POR VAN)
        base_dist_vals_avg_van = [s["baseline_distance"] for s in sims]
        ga_dist_vals_avg_van = [s["ga_distance"] for s in sims]
        diff_dist_vals_avg_van = [g - b for g, b in zip(ga_dist_vals_avg_van, base_dist_vals_avg_van)]

        def fmt_time(vals):
            m = mean(vals)
            s = safe_stdev(vals)
            return f"{minutes_to_Hh_Mm_string(m)} (± {int(s)}m)"

        def fmt_dist(vals):
            m = mean(vals)
            s = safe_stdev(vals)
            m_str = meters_to_km_string(m)
            s_km = s / 1000.0
            return f"{m_str} (± {s_km:.1f} km)"

        rows.append({
            "Pacientes": num_pat,
            "Vans": vans,
            "Tempo/Passageiro Baseline": fmt_time(base_cost_vals_avg_pax),
            "Tempo/Passageiro GA": fmt_time(ga_cost_vals_avg_pax),
            "Dif. Tempo (GA - Base) / Passageiro": fmt_time(diff_cost_vals_avg_pax),
            "Distância/Van Baseline": fmt_dist(base_dist_vals_avg_van),
            "Distância/Van GA": fmt_dist(ga_dist_vals_avg_van),
            "Dif. Dist. (GA - Base) / Van": fmt_dist(diff_dist_vals_avg_van)
        })

    df = pd.DataFrame(rows)
    return df, results

def get_api_key_from_file(filepath: str) -> str:
    """
    Lê a API Key do Google Maps de um arquivo de texto simples.
    O arquivo deve conter apenas a chave.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        key = f.read().strip()
    return key


###############################################
# EXEMPLO COMPLETO DE USO
###############################################

if __name__ == "__main__":

    API_KEY_FILE = "./google_api_key.txt"
    API_KEY = get_api_key_from_file(API_KEY_FILE)
    CACHE_FILENAME = './teste.json'
    
    # Tentativa de carregamento dos Managers e Loader
    try:
        # Assumindo que DistanceManager e AddressLoader estão disponíveis
        google_dm = DistanceManager(api_key=API_KEY, cache_filepath=CACHE_FILENAME)
        dm = RealDistanceManager(google_dm)
        from addresses import AddressLoader
        loader = AddressLoader("./enderecos.txt") # Assumindo arquivo de endereços
        
        # Para a opção 3, garantir o loader com dados. 
        # Se for um teste de Dummy, esta seção deve ser adaptada.
        if EXECUTION_MODE == 3:
             loader_maps = AddressLoader("./enderecos.txt")
        else:
             loader_maps = None

    except NameError:
        print("Usando Dummy para demonstração. Não é possível rodar a Opção 3 sem AddressLoader.")
        dm = DummyDistanceManager()
        class DummyLoader:
            def __init__(self, filename=None): pass
            def get_random_address(self):
                return {'city': 'Dummy City', 'address': 'Dummy Street', 'uf': 'DU'}
            def get_all(self):
                # Retorna dados mock para teste da opção 3 (não deve ser usado para geocodificação)
                return [{'address': 'Rua Mock 1, Cidade D', 'city': 'Dummy City', 'uf': 'DU'},
                        {'address': 'Rua Mock 2, Cidade D', 'city': 'Dummy City', 'uf': 'DU'}]
        loader = DummyLoader()
        loader_maps = DummyLoader()


    start_address = Address.from_real_address({'address': 'Praça Barão do Rio Branco, 12 - Pilar, Ouro Preto - MG', 'city': 'Ouro Preto', 'uf': 'MG'})

    if EXECUTION_MODE == 1:
        # --- OPÇÃO 1: RODAR TABELA ---
        print("\n=== MODO 1: RODAR TABELA DE SIMULAÇÕES ===")
        ENABLE_DETAILED_SUMMARY = False # Desativa sumários detalhados
        
        df, raw = simulate_and_build_table_df(
            patient_range=range(5, 11, 5), # Ex: 5 e 10 pacientes
            van_range=range(2, 5), # Alterado: range de vans (Ex: 2, 3 e 4 vans)
            n_runs=5, # poucas rodadas para teste rápido
            dm=dm,
            loader=loader,
            start_address=start_address,
            simulation_fn=run_random_simulation
        )

        print("\n=== TABELA DE RESULTADOS AGREGADOS ===")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df)

        df.to_csv("simulation_results.csv", index=True)

    elif EXECUTION_MODE == 2:
        # --- OPÇÃO 2: RODAR UMA SIMULAÇÃO ALEATÓRIA ---
        print("\n=== MODO 2: RODAR SIMULAÇÃO ALEATÓRIA ÚNICA ===")
        ENABLE_DETAILED_SUMMARY = True # Ativa sumários detalhados
        
        run_random_simulation(
            dm=dm,
            loader=loader,
            num_patients=5,
            num_vans=3,
            start_address=start_address,
            save_summaries=True
        )
        print("\nSimulação concluída com sumário detalhado.")

    elif EXECUTION_MODE == 3:
        # --- OPÇÃO 3: ABRIR TODOS OS PONTOS NO GOOGLE MAPS ---
        print("\n=== MODO 3: ABRIR TODOS OS ENDEREÇOS NO GOOGLE MAPS ===")
        ENABLE_DETAILED_SUMMARY = False # Desativa sumários (não se aplica, mas por segurança)
        
        try:
            enderecos = loader_maps.get_all()
            
            # Formatação para o Google Maps /dir/
            partes_da_url = []
            for e in enderecos:
                 # Remove espaços e vírgulas para uma URL limpa (simplificação do get_address_string)
                partes_da_url.append(e['address'].replace(" ", "+").replace(",", ""))

            # Criação da URL final
            url_google_maps = f"https://www.google.com/maps/dir/{'/'.join(partes_da_url)}"

            print(f"A URL gerada é: {url_google_maps}")
            webbrowser.open(url_google_maps)

        except Exception as e:
            print(f"Erro ao tentar abrir o Google Maps: {e}. Certifique-se de que AddressLoader e 'enderecos.txt' estão configurados corretamente.")
            print("Se estiver usando a simulação Dummy, a URL gerada não será válida para o Google Maps.")

    else:
        print("Modo de execução inválido. Defina EXECUTION_MODE para 1, 2 ou 3.")