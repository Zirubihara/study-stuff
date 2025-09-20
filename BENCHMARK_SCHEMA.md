# Benchmark System Architecture Schema

## 📂 Struktura Katalogów

```
📁 scripts/benchmarks/
│
├── 📁 dataset_specific/           # Skrypty orkiestrujące dla konkretnych rozmiarów
│   ├── benchmark_5million_dataset.py    ← ORKIESTRATOR 5M
│   ├── benchmark_10million_dataset.py   ← ORKIESTRATOR 10M
│   ├── benchmark_1m_10m_comparison.py   ← ORKIESTRATOR 1M+10M
│   ├── benchmark_50million_dataset.py   ← ORKIESTRATOR 50M
│   └── benchmark_100million_dataset.py  ← ORKIESTRATOR 100M
│
├── 📁 implementations/           # Bazowe implementacje technologii
│   ├── benchmark_pandas_implementation.py   ← PANDAS ENGINE
│   ├── benchmark_polars_implementation.py   ← POLARS ENGINE
│   ├── benchmark_pyarrow_implementation.py  ← PYARROW ENGINE
│   ├── benchmark_dask_implementation.py     ← DASK ENGINE
│   └── benchmark_pyspark_implementation.py  ← PYSPARK ENGINE
│
└── 📁 runners/                   # Zaawansowane orkiestratory
    ├── benchmark_runner_comprehensive.py
    └── benchmark_library_comparison.py
```

## 🔄 Przepływ Wykonania

### Poziom 1: WYWOŁANIE GŁÓWNE
```bash
cd scripts/benchmarks/dataset_specific
python benchmark_10million_dataset.py
```

### Poziom 2: ORKIESTRATOR (benchmark_10million_dataset.py)
```python
def main():
    # 1. SPRAWDZENIE DANYCH
    dataset_path = Path("../../../data/benchmark_10m.csv")
    if not dataset_path.exists(): ERROR

    # 2. DEFINICJA TECHNOLOGII
    technologies = {
        "pandas":  Path("../implementations/benchmark_pandas_implementation.py"),
        "polars":  Path("../implementations/benchmark_polars_implementation.py"),
        "pyarrow": Path("../implementations/benchmark_pyarrow_implementation.py"),
        "dask":    Path("../implementations/benchmark_dask_implementation.py"),
        "pyspark": Path("../implementations/benchmark_pyspark_implementation.py")
    }

    # 3. WYKONANIE DLA KAŻDEJ TECHNOLOGII
    for tech_name, script_path in technologies.items():
        run_technology(script_path, tech_name)
```

### Poziom 3: WYKONANIE TECHNOLOGII (run_technology)
```python
def run_technology(script_path, technology_name):
    # 1. MODYFIKACJA SKRYPTU
    temp_script = modify_script_for_10m(script_path, technology_name)

    # 2. URUCHOMIENIE SUBPROCESS
    result = subprocess.run([python, temp_script])

    # 3. CZYSZCZENIE
    temp_script.unlink()  # Usuwa tymczasowy plik
```

### Poziom 4: MODYFIKACJA KONFIGURACJI (modify_script_for_10m)
```python
def modify_script_for_10m(script_path, technology_name):
    # CZYTA oryginalny plik implementacji
    with open(script_path, "r") as f:
        content = f.read()

    # ZAMIENIA konfigurację datasetu:
    replacements = [
        ('csv_path = "data/benchmark_5m.csv"',
         'csv_path = "../../../data/benchmark_10m.csv"'),
        ('output_path = "../results/performance_metrics_pandas.json"',
         'output_path = "../../../results/performance_metrics_pandas_10m.json"')
    ]

    for old, new in replacements:
        content = content.replace(old, new)

    # TWORZY tymczasowy plik
    temp_script = Path(f"temp_{technology_name}_10m.py")
    with open(temp_script, "w") as f:
        f.write(content)

    return temp_script
```

### Poziom 5: IMPLEMENTACJA TECHNOLOGII (np. benchmark_pandas_implementation.py)
```python
def main():
    # KONFIGURACJA (zostanie zmodyfikowana przez orkiestrator)
    csv_path = "data/benchmark_5m.csv"  # ← ZOSTANIE ZAMIENIONE NA 10M
    output_path = "../results/performance_metrics_pandas.json"  # ← ZOSTANIE ZAMIENIONE

    # RZECZYWISTE PRZETWARZANIE
    processor = PandasDataProcessor(csv_path)
    results = processor.process_data()
    processor.save_performance_metrics(output_path)
```

## 🗂️ Przepływ Plików

```
WYWOŁANIE:
benchmark_10million_dataset.py

↓ MODYFIKUJE ↓

ORYGINAŁ:                           TYMCZASOWY:
benchmark_pandas_implementation.py → temp_pandas_10m.py
benchmark_polars_implementation.py → temp_polars_10m.py
benchmark_pyarrow_implementation.py → temp_pyarrow_10m.py
benchmark_dask_implementation.py → temp_dask_10m.py
benchmark_pyspark_implementation.py → temp_pyspark_10m.py

↓ URUCHAMIA SUBPROCESS ↓

KAŻDY TYMCZASOWY PLIK:
1. CZYTA: ../../../data/benchmark_10m.csv
2. PRZETWARZA: dane przez swoją technologię
3. ZAPISUJE: ../../../results/performance_metrics_<tech>_10m.json

↓ CZYŚCI ↓

USUWA wszystkie pliki temp_*_10m.py
```

## 🔧 Kluczowe Mechanizmy

### 1. SYSTEM ŚCIEŻEK
```
WZGLĘDEM: scripts/benchmarks/dataset_specific/
├── ../implementations/           (implementacje)
├── ../../../data/               (dane wejściowe)
└── ../../../results/            (wyniki)
```

### 2. SYSTEM NAZEWNICTWA
```
WZORZEC PLIKÓW:
- Orkiestrator: benchmark_<SIZE>_dataset.py
- Implementacja: benchmark_<TECH>_implementation.py
- Tymczasowy: temp_<TECH>_<SIZE>.py
- Wynik: performance_metrics_<TECH>_<SIZE>.json
```

### 3. SYSTEM MODYFIKACJI
```python
ZAMIENIANE ELEMENTY:
1. csv_path        → ścieżka do datasetu
2. output_path     → ścieżka do wyników
3. komentarze      → aktualizacja rozmiaru datasetu
```

## ⚠️ Ważne Punkty

1. **ORKIESTRATORY nie wykonują obliczeń** - tylko zarządzają procesem
2. **IMPLEMENTACJE są modyfikowane w locie** - oryginalny plik pozostaje niezmieniony
3. **TYMCZASOWE PLIKI są usuwane** po wykonaniu
4. **KAŻDA TECHNOLOGIA działa w osobnym subprocess** - izolacja błędów
5. **ŚCIEŻKI SĄ WZGLĘDNE** od miejsca uruchomienia orkiestratora

## 🎯 Przykład Użycia

```bash
# Uruchom benchmark 10M dla wszystkich technologii
cd scripts/benchmarks/dataset_specific
python benchmark_10million_dataset.py

# Rezultat:
# - 5 plików JSON w results/ z metrykami wydajności
# - Każda technologia przetworzyła dokładnie 10,000,000 wierszy
# - Porównywalne wyniki między technologiami
```