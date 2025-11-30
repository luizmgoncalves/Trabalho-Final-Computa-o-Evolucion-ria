import googlemaps
import json
import os
import pprint

class DistanceManager:
    """
    Gerencia o cálculo e o cache de matrizes de distância usando a API do Google Maps.

    Esta classe otimiza as chamadas de API ao:
    1. Usar um cache em memória para evitar consultas repetidas.
    2. Carregar e salvar este cache em um arquivo JSON para persistência entre execuções.
    3. Aproximar distâncias entre cidades diferentes consultando a rota apenas
       entre as cidades, e não entre os endereços específicos.
    """
    def __init__(self, api_key: str, points: list = None, cache_filepath: str = 'distance_cache.json'):
        """
        Inicializa o gerenciador de distâncias.

        Args:
            api_key (str): Sua chave de API do Google Cloud.
            points (list, optional): Uma lista inicial de dicionários para cálculo da matriz.
                                     Ex: [{'address': '...', 'city': '...'}, ...]. Defaults to None.
            cache_filepath (str): O caminho do arquivo para carregar e salvar o cache.
        """
        if not api_key or api_key == 'SUA_CHAVE_DE_API':
            raise ValueError("Por favor, forneça uma chave de API válida do Google Maps.")
        
        self.points = points if points is not None else []
        self.api_key = api_key
        self.cache_filepath = cache_filepath
        self.gmaps = googlemaps.Client(key=self.api_key)
        
        self.cache = self._load_from_file()
        if self.points:
            self.matrix = [[0.0] * len(self.points) for _ in range(len(self.points))]
        else:
            self.matrix = []

    def _load_from_file(self) -> dict:
        """Carrega o cache de distâncias de um arquivo JSON, se existir."""
        if os.path.exists(self.cache_filepath):
            # print(f"INFO: Encontrado arquivo de cache em '{self.cache_filepath}'. Carregando distâncias...")
            try:
                with open(self.cache_filepath, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                # print(f"AVISO: Não foi possível ler o arquivo de cache. Começando com um cache vazio. Erro: {e}")
                return {}
        else:
            # print("INFO: Nenhum arquivo de cache encontrado. O cache começará vazio.")
            return {}

    def save_cache(self):
        """Salva o cache de distâncias atual em um arquivo JSON."""
        # print(f"INFO: Salvando cache com {len(self.cache)} entradas em '{self.cache_filepath}'...")
        try:
            with open(self.cache_filepath, 'w') as f:
                json.dump(self.cache, f, indent=4)
            # print("INFO: Cache salvo com sucesso.")
        except IOError as e:
            print(f"ERRO: Não foi possível salvar o arquivo de cache. Erro: {e}")

    def _query_api(self, origin: str, destination: str) -> float:
        """Consulta a API do Google para uma única distância e retorna o valor em metros."""
         #print(f"API CALL: Consultando distância de '{origin}' para '{destination}'...")
        try:
            matrix = self.gmaps.distance_matrix(origin, destination, mode="driving")
            element = matrix['rows'][0]['elements'][0]
            if element['status'] == 'OK':
                return float(element['distance']['value'])
            else:
                print(f"AVISO: Rota não encontrada entre '{origin}' e '{destination}'. Status: {element['status']}")
                return float('inf')
        except googlemaps.exceptions.ApiError as e:
            # print(f"ERRO DE API: {e}")
            return float('inf')

    def _get_distance_logic(self, origin_point: dict, destination_point: dict) -> float:
        """Lógica central para obter a distância, usando cache e otimização por cidade."""
        if origin_point['address'] == destination_point['address']:
            return 0.0

        if origin_point['city'] == destination_point['city']:
            origin_api = origin_point['address']
            destination_api = destination_point['address']
            cache_key = f"{origin_api} | {destination_api}"
        else:
            origin_api = origin_point['city']
            destination_api = destination_point['city']
            cache_key = f"{origin_api} | {destination_api}"

        if cache_key in self.cache:
            # print(f"CACHE HIT: De '{origin_api}' para '{destination_api}'.")
            return self.cache[cache_key]

        distance = self._query_api(origin_api, destination_api)
        self.cache[cache_key] = distance
        return distance
    
    def get_distance_between_points(self, point1: dict, point2: dict) -> float:
        """
        Consulta a distância específica entre dois pontos fornecidos sob demanda.

        Utiliza a mesma lógica de cache e otimização por cidade da classe.
        Os pontos devem estar no formato {'address': '...', 'city': '...'}.

        Args:
            point1 (dict): O ponto de origem.
            point2 (dict): O ponto de destino.

        Returns:
            float: A distância em metros, ou float('inf') se não for encontrada.
        """
        if not all(k in point1 for k in ['address', 'city']) or \
           not all(k in point2 for k in ['address', 'city']):
            raise ValueError("Os pontos devem ser dicionários contendo as chaves 'address' e 'city'.")
        
        return self._get_distance_logic(point1, point2)

    # Alias para o método, como solicitado
    distancia = get_distance_between_points

    def compute_matrix(self) -> list:
        """
        Calcula e retorna a matriz de distâncias cruzadas completa para os pontos da inicialização.
        """
        if not self.points:
            # print("AVISO: Nenhum ponto foi fornecido na inicialização. A matriz está vazia.")
            return []
            
        # print("\n--- Iniciando cálculo da matriz de distâncias ---")
        num_points = len(self.points)
        for i in range(num_points):
            for j in range(num_points):
                if i == j:
                    continue
                
                origin = self.points[i]
                destination = self.points[j]
                self.matrix[i][j] = self._get_distance_logic(origin, destination)
        
        # print("--- Cálculo da matriz finalizado ---\n")
        return self.matrix
    
    def get_matrix_from_addresses(self, address_list):
        """
        Recebe uma lista de objetos Address (do modelo da otimização) e retorna
        uma matriz de distâncias na mesma ordem usando a lógica real de distâncias.

        Cada Address.raw deve ser um dict do tipo:
            {'address': '...', 'city': '...'}

        Retorna:
            list[list[float]]
        """
        # Extrai dicts reais dos objetos Address
        points = []
        for addr in address_list:
            if addr.raw is None:
                raise ValueError(
                    "Address recebido não contém endereço real (.raw). "
                    "Use Address.from_real_address({'address':..., 'city':...})"
                )
            points.append(addr.raw)

        n = len(points)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                matrix[i][j] = self._get_distance_logic(points[i], points[j])

        return matrix


def get_api_key_from_file(filepath: str) -> str:
    """
    Lê a API Key do Google Maps de um arquivo de texto simples.
    O arquivo deve conter apenas a chave.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        key = f.read().strip()
    return key


# --- Exemplo de Uso ---
if __name__ == "__main__":
    # Substitua pela sua chave de API
    API_KEY_FILE = "./google_api_key.txt"
    API_KEY = get_api_key_from_file(API_KEY_FILE)
    CACHE_FILENAME = './distancias_salvas.json'

    # Lista de pontos de interesse
    pontos_de_interesse = [
        {'address': 'Avenida Paulista, 1578, São Paulo, SP', 'city': 'São Paulo'}, # 0
        {'address': 'Parque Ibirapuera, São Paulo, SP', 'city': 'São Paulo'},      # 1
        {'address': 'Praia de Copacabana, Rio de Janeiro, RJ', 'city': 'Rio de Janeiro'}, # 2
    ]

    try:
        # --- ETAPA 1: Calcular a matriz inicial e salvar no cache ---
        print("="*50)
        print("ETAPA 1: CÁLCULO INICIAL DA MATRIZ")
        print("="*50)
        manager = DistanceManager(api_key=API_KEY, points=pontos_de_interesse, cache_filepath=CACHE_FILENAME)
        matriz_inicial = manager.compute_matrix()
        print("Matriz Inicial Calculada:")
        pprint.pprint(matriz_inicial)
        manager.save_cache() # Salva o que foi aprendido

        print("\n\n")

        # --- ETAPA 2: Usar o método 'distancia' para consultas específicas ---
        print("="*50)
        print("ETAPA 2: CONSULTAS ESPECÍFICAS (AD-HOC)")
        print("="*50)
        
        # Ponto 1: Consulta que já deve estar no cache por causa do cálculo da matriz
        print("--> Consulta 1: Entre pontos já conhecidos (deve usar cache)")
        p1 = {'address': 'Avenida Paulista, 1578, São Paulo, SP', 'city': 'São Paulo'}
        p2 = {'address': 'Parque Ibirapuera, São Paulo, SP', 'city': 'São Paulo'}
        dist = manager.distancia(p1, p2)
        print(f"Distância entre Av. Paulista e Ibirapuera: {dist/1000:.2f} km\n")

        # Ponto 2: Consulta de um ponto conhecido para um ponto NOVO
        print("--> Consulta 2: Ponto conhecido para um ponto novo (deve usar API)")
        p3 = {'address': 'Aeroporto de Congonhas, São Paulo, SP', 'city': 'São Paulo'}
        dist2 = manager.distancia(p2, p3)
        print(f"Distância entre Ibirapuera e Congonhas: {dist2/1000:.2f} km\n")

        # Ponto 3: Repetir a consulta anterior (agora deve usar o cache)
        print("--> Consulta 3: Repetindo a consulta anterior (deve usar cache)")
        dist3 = manager.distancia(p2, p3)
        print(f"Distância entre Ibirapuera e Congonhas (repetido): {dist3/1000:.2f} km\n")
        
        # Ponto 4: Consulta entre cidades que já está no cache pela otimização
        print("--> Consulta 4: Entre cidades diferentes já cacheadas")
        p4 = {'address': 'Pão de Açúcar, Rio de Janeiro, RJ', 'city': 'Rio de Janeiro'}
        dist4 = manager.distancia(p1, p4) # Da Av. Paulista para o Pão de Açúcar
        print(f"Distância entre São Paulo e Rio de Janeiro: {dist4/1000:.2f} km\n")

        # --- ETAPA 3: Salvar o cache atualizado ---
        # Note que o cache agora contém as distâncias de Congonhas
        print("="*50)
        print("ETAPA 3: SALVANDO O CACHE ATUALIZADO")
        print("="*50)
        manager.save_cache()

    except ValueError as e:
        print(f"\nERRO DE CONFIGURAÇÃO: {e}\nPor favor, defina a variável API_KEY.")