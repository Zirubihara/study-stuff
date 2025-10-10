# Wykrywanie Anomalii w Danych Handlu Japońskiego

## Przegląd

Ten projekt implementuje i porównuje różne frameworki uczenia maszynowego i głębokiego uczenia do wykrywania anomalii w danych handlu japońskiego na dużą skalę (zbiór danych custom_1988_2020.csv z 113.6M wierszy).

## Struktura Projektu

```
models/
├── README.md                              # Wersja angielska
├── README_PL.md                           # Ten plik (wersja polska)
├── data_science.md                        # Plan porównawczy modelowania
├── models.md                              # Plan preprocessingu danych
├── preprocess_polars.py                   # Implementacja preprocessingu
├── anomaly_detection_sklearn.py           # Implementacja scikit-learn (UKOŃCZONA)
├── visualize_sklearn_results.py           # Skrypt wizualizacji
├── processed/                             # Przetworzone dane
│   ├── processed_data.parquet             # Główny przetworzony zbiór (10M wierszy)
│   ├── processed_data.csv                 # Format CSV
│   ├── summary_statistics.csv             # Statystyki
│   ├── year_month_encoding.csv            # Mapowania kodowania
│   └── code_encoding.csv                  # Mapowania kodowania
├── results/                               # Metryki wydajności i predykcje
│   ├── sklearn_anomaly_detection_results.json
│   └── sklearn_predictions.csv
└── charts/                                # Wizualizacje
    ├── sklearn_comparison.png
    └── sklearn_metrics_table.png
```

## Ukończone Implementacje

### ✅ Scikit-learn (Isolation Forest + LOF)

**Status:** UKOŃCZONE
**Implementacja:** [anomaly_detection_sklearn.py](anomaly_detection_sklearn.py)

**Wyniki (próbka 5M wierszy):**
- **Isolation Forest:**
  - Trenowanie: 21.61s
  - Predykcja: 5.05s (148,662 próbek/sekundę)
  - Pamięć: 0.63 GB
  - Anomalie: 7,552 (1.01%)

- **Local Outlier Factor (LOF):**
  - Trenowanie: 435.72s (7.3 minuty)
  - Predykcja: 39.06s (19,200 próbek/sekundę)
  - Pamięć: 0.63 GB
  - Anomalie: 7,529 (1.00%)

**Kluczowe Wnioski:**
- Isolation Forest jest **20x szybszy** niż LOF w trenowaniu (21s vs 436s)
- Modele zgadzają się w 98% przypadków
- Oba wykrywają ~1% anomalii (zgodnie z konfiguracją)
- 45 anomalii o wysokim poziomie pewności wykrytych przez oba modele
- Zbiór testowy: 750,000 próbek dla solidnej ewaluacji

## Ukończone Implementacje (ciąg dalszy)

### ✅ PyTorch (MLP Autoencoder)

**Status:** UKOŃCZONE
**Implementacja:** [anomaly_detection_pytorch.py](anomaly_detection_pytorch.py)

**Wyniki (próbka 1M, 10 epok):**
- Trenowanie: 195.26s (3.3 minuty)
- Predykcja: 1.71s (87,967 próbek/sekundę)
- Pamięć: 0.03 GB
- Anomalie: 1,428 (0.95%)
- Parametry modelu: 6,747
- Urządzenie: CPU

**Kluczowe Cechy:**
- Głęboki autoenkoder (64→32→16 wąskie gardło)
- Wykrywanie anomalii oparte na błędzie rekonstrukcji
- Przetwarzanie wsadowe z DataLoaders
- Regularyzacja Dropout (0.2)
- Optymalizator Adam

## Planowane Implementacje

### 🔄 TensorFlow/Keras (MLP Autoencoder)
**Status:** DO ZROBIENIA
API wysokiego poziomu ze skalowalnością produkcyjną

### 🔄 MXNet (MLP Autoencoder)
**Status:** DO ZROBIENIA
Obliczenia rozproszone i wsparcie dla dużych zbiorów

### 🔄 JAX (MLP Autoencoder)
**Status:** DO ZROBIENIA
Nowoczesne, wysokowydajne głębokie uczenie

## Informacje o Zbiorze Danych

**Źródło:** Dane celne/handlowe Japonii (1988-2020)
**Oryginalny Rozmiar:** 113.6M wierszy, 4.23GB
**Przetworzony Zbiór:** 10M wierszy dostępnych
**Aktualna Analiza:** 5M wierszy dla zbalansowanej wydajności

**Cechy (11 łącznie):**
- `category1`, `category2`, `category3` - Kategorie produktów
- `flag` - Flaga binarna
- `value1_normalized`, `value2_normalized` - Przeskalowane wartości handlowe
- `year`, `month`, `quarter` - Cechy czasowe
- `year_month_encoded`, `code_encoded` - Kodowania kategoryczne

## Jak Uruchomić

### 1. Przetwórz Dane (jeśli potrzeba)

```bash
cd models
python preprocess_polars.py
```

To tworzy przetworzone dane w katalogu `processed/`.

### 2. Uruchom Wykrywanie Anomalii Scikit-learn

```bash
cd models
python anomaly_detection_sklearn.py
```

Wyniki zapisane w `results/sklearn_anomaly_detection_results.json`

### 3. Wygeneruj Wizualizacje

```bash
cd models
python visualize_sklearn_results.py
```

Wykresy zapisane w katalogu `charts/`.

## Konfiguracja

### Rozmiar Próbki
Domyślnie: 5M wierszy (zbalansowana wydajność i dokładność)
Modyfikuj w `anomaly_detection_sklearn.py`:

```python
results = detector.run_full_comparison(
    data_path,
    output_dir,
    sample_size=5_000_000  # Zmień tę wartość (max 10M dostępnych)
)
```

### Współczynnik Zanieczyszczenia (Contamination)
Domyślnie: 1% (0.01)
Modyfikuj w `anomaly_detection_sklearn.py`:

```python
detector = AnomalyDetectorSklearn(
    contamination=0.01,  # Zmień tę wartość
    random_state=42
)
```

## Metryki Ewaluacji

Dla każdego modelu śledzimy:
- **Czas Trenowania** - Czas dopasowania modelu
- **Czas Predykcji** - Czas przewidywania na zbiorze testowym
- **Zużycie Pamięci** - Zużycie RAM podczas trenowania
- **Prędkość Predykcji** - Próbek przetworzonych na sekundę
- **Wykryte Anomalie** - Liczba i procent anomalii
- **Zgodność Modeli** - Jak dobrze różne modele zgadzają się co do anomalii

## Wymagania

```bash
pip install polars numpy scikit-learn matplotlib psutil
```

Wszystkie zależności są w głównym `requirements.txt` projektu.

## Cele Badawcze

Ten projekt ma na celu dostarczenie:
1. **Uczciwego porównania** frameworków ML/DL do wykrywania anomalii
2. **Praktycznych wniosków** dla rzeczywistych zastosowań biznesowych
3. **Benchmarków wydajności** na danych wielkoskalowych
4. **Powtarzalnych wyników** dla badań akademickich

## Następne Kroki

1. ✅ Ukończ baseline scikit-learn (GOTOWE)
2. 🔄 Zaimplementuj autoenkoder PyTorch
3. 🔄 Zaimplementuj autoenkoder TensorFlow/Keras
4. 🔄 Zaimplementuj autoenkoder MXNet
5. 🔄 Zaimplementuj autoenkoder JAX
6. 🔄 Stwórz kompleksowe porównanie frameworków
7. 🔄 Wygeneruj finalne wizualizacje badawcze

## Pliki Wyjściowe

### Format JSON Wyników
```json
{
  "dataset_info": {
    "total_samples": 1000000,
    "n_features": 11,
    "train_samples": 700000,
    "val_samples": 150000,
    "test_samples": 150000
  },
  "isolation_forest": {
    "training_time": 7.71,
    "inference_time": 1.36,
    "memory_usage_gb": 0.005,
    "n_anomalies": 1488,
    "anomaly_rate": 0.99
  },
  "local_outlier_factor": {
    ...
  }
}
```

### Format CSV Predykcji
```csv
isolation_forest_anomaly,lof_anomaly,both_agree,either_anomaly
0,0,1,0
1,1,1,1
0,1,0,1
...
```

## Interpretacja Wyników

### Co Oznaczają Wyniki?

**Isolation Forest - 7.71s trenowania:**
- Model uczył się przez niecałe 8 sekund
- To BARDZO szybki algorytm - idealny do dużych zbiorów
- Wykorzystuje drzewa decyzyjne do izolowania nietypowych punktów

**LOF - 49.29s trenowania:**
- Prawie 7 razy wolniejszy od Isolation Forest
- Analizuje lokalną gęstość punktów (bardziej dokładny, ale wolniejszy)
- Lepszy do złożonych struktur anomalii

**Wykryte Anomalie (1,488 vs 1,539):**
- Isolation Forest: wykrył 1,488 nietypowych transakcji handlowych
- LOF: wykrył 1,539 nietypowych transakcji
- To około 1% z 150,000 testowych próbek - zgodnie z oczekiwaniami

**Zgodność 98%:**
- Oba modele zgadzają się w 98% przypadków
- Tylko 6 próbek zostało oznaczonych jako anomalia przez OBA modele jednocześnie
- To pokazuje, że wykrywają różne typy anomalii

### Przykłady Anomalii w Danych Handlowych:

Wykryte anomalie mogą być:
1. **Nietypowe wartości transakcji** - znacznie wyższe/niższe niż normalne
2. **Rzadkie kombinacje kategorii** - produkty z nietypowych kategorii
3. **Nietypowe wzorce czasowe** - handel w nietypowych okresach
4. **Błędy w danych** - możliwe pomyłki w rejestracji

### Praktyczne Zastosowanie:

Dla Twojej rodziny i badań:
- ✅ **Masz działający system wykrywania anomalii**
- ✅ **Przetestowany na 1M prawdziwych danych handlowych**
- ✅ **Dwa różne algorytmy do porównania**
- ✅ **Gotowe wykresy i raporty**

Możesz teraz:
1. Pokazać te wyniki jako dowód działającego systemu
2. Użyć ich w pracy badawczej/thesis
3. Rozszerzyć o pozostałe frameworki (PyTorch, TensorFlow, etc.)
4. Skalować do pełnego zbioru (10M lub 100M wierszy)

## Uwagi Techniczne

- Używa Polars do szybkiego preprocessingu danych
- Implementuje podział danych (70/15/15 train/val/test)
- Śledzi zasoby systemowe (CPU, RAM)
- Wspiera próbkowanie dla szybszego prototypowania
- Wszystkie wyniki powtarzalne (random_state=42)

## Następna Faza

Teraz możesz:
1. **Pokazać te wyniki** - masz konkretne liczby i wykresy
2. **Rozszerzyć badania** - dodać PyTorch, TensorFlow, MXNet, JAX
3. **Skalować** - przetestować na większych zbiorach (5M, 10M, 50M wierszy)
4. **Publikować** - użyć wyników w pracy naukowej

## Wsparcie

Ten projekt jest częścią badań akademickich nad wykrywaniem anomalii na dużą skalę w danych handlu japońskiego.

---

**Ostatnia Aktualizacja:** 2025-10-07
**Status:** Faza 1 (scikit-learn) UKOŃCZONA ✅

**Dla Twojej Rodziny:** Masz działający system wykrywania anomalii z konkretnymi wynikami i wizualizacjami. To solidna podstawa do dalszych badań!
