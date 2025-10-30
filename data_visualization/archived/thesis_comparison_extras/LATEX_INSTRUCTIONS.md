# 📝 Jak Użyć Kodu w Listingach LaTeX

## 🎯 Masz Teraz 3 Pliki:

### 1. **CLEAN_CODE_FOR_LATEX.py** (Python)
- ✅ Czyste, działające przykłady kodu
- ✅ Minimalne, ale funkcjonalne
- ✅ Dobrze udokumentowane
- **Użyj:** Jako źródło do kopiowania

### 2. **LATEX_LISTINGS_READY.tex** (LaTeX)
- ✅ Gotowe listingi z `\begin{lstlisting}...\end{lstlisting}`
- ✅ Labels i captions już ustawione
- ✅ Przykłady użycia w tekście
- **Użyj:** Skopiuj bezpośrednio do pracy

### 3. **LATEX_CODE_LISTINGS.tex** (LaTeX - Pełny Dokument)
- ✅ Kompletny dokument z analizą
- ✅ 50+ stron gotowej treści
- **Użyj:** Jako inspiracja do rozdziału

---

## 🚀 Quick Start - 3 Kroki

### KROK 1: Dodaj Preambułę do Swojej Pracy

W `main.tex` dodaj przed `\begin{document}`:

```latex
% Pakiety dla kodu
\usepackage{listings}
\usepackage{xcolor}

% Kolory dla składni Pythona
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

% Styl dla Pythona
\lstdefinestyle{pythonstyle}{
    language=Python,
    backgroundcolor=\color{backcolour},
    commentstyle=\color{codegreen},
    keywordstyle=\color{blue}\bfseries,
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breaklines=true,
    numbers=left,
    numbersep=5pt,
    frame=single,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=2
}

\lstset{style=pythonstyle}
```

### KROK 2: Skopiuj Listing do Rozdziału

Otwórz `LATEX_LISTINGS_READY.tex` i skopiuj listing:

```latex
\section{Porównanie Bibliotek}

\subsection{Plotly - Najkrótsza Implementacja}

Listing~\ref{lst:plotly_simple} przedstawia implementację w Plotly Express,
charakteryzującą się najkrótszym kodem.

% SKOPIUJ STĄD (z LATEX_LISTINGS_READY.tex):
\begin{lstlisting}[caption={Plotly Express - 8 LOC},label={lst:plotly_simple}]
import plotly.express as px

df = pd.DataFrame({
    'Library': ['Pandas', 'Polars', 'PyArrow', 'Dask', 'Spark'],
    'Time': [11.0, 1.51, 5.31, 22.28, 99.87]
})

fig = px.bar(df, x='Library', y='Time', color='Library')
fig.write_html('output.html')
\end{lstlisting}

Jak widać w Listing~\ref{lst:plotly_simple}, Plotly wymaga zaledwie
8 linii kodu, podczas gdy Bokeh (Listing~\ref{lst:bokeh_simple}) 
wymaga 25 linii dla identycznej funkcjonalności.
```

### KROK 3: Kompiluj

```bash
pdflatex main.tex
```

---

## 📊 Przykłady Użycia w Pracy

### Przykład 1: Prosty Listing z Analizą

```latex
\section{Analiza Złożoności Kodu}

\subsection{Wykres Słupkowy}

Implementacja prostego wykresu słupkowego w różnych bibliotekach 
wykazuje znaczące różnice w ilości wymaganego kodu.

\begin{lstlisting}[caption={Plotly - 8 linii},label={lst:plotly}]
import plotly.express as px
df = pd.DataFrame({'Library': [...], 'Time': [...]})
fig = px.bar(df, x='Library', y='Time')
fig.write_html('output.html')
\end{lstlisting}

Listing~\ref{lst:plotly} pokazuje, że Plotly wymaga jedynie 8 linii 
kodu. Dla porównania, implementacja w Bokeh wymaga 25 linii -- 
\textbf{3-krotnie więcej}.

\textbf{Główne różnice:}
\begin{itemize}
    \item Plotly: Deklaratywne API (2 linie = wykres)
    \item Bokeh: Imperatywne API (15 linii = wykres + styling)
\end{itemize}
```

### Przykład 2: Side-by-Side Comparison

```latex
\subsection{Porównanie Składni}

\begin{figure}[h]
\begin{minipage}{0.48\textwidth}
\begin{lstlisting}[caption=Plotly,basicstyle=\ttfamily\small]
# Plotly - 2 linie
fig = px.bar(df, x='x', y='y')
fig.write_html('out.html')
\end{lstlisting}
\end{minipage}
\hfill
\begin{minipage}{0.48\textwidth}
\begin{lstlisting}[caption=Bokeh,basicstyle=\ttfamily\small]
# Bokeh - 10+ linii
source = ColumnDataSource(...)
p = figure(...)
p.vbar(x='x', top='y', ...)
hover = HoverTool(...)
p.add_tools(hover)
output_file('out.html')
save(p)
\end{lstlisting}
\end{minipage}
\caption{Porównanie: Plotly vs Bokeh}
\end{figure}
```

### Przykład 3: Tabela + Listing

```latex
\subsection{Wyniki Porównania}

Tabela~\ref{tab:loc} przedstawia średnią liczbę linii kodu dla każdej 
biblioteki.

\begin{table}[h]
\centering
\caption{Średnia Lines of Code (LOC)}
\label{tab:loc}
\begin{tabular}{|l|c|}
\hline
\textbf{Biblioteka} & \textbf{Śr. LOC} \\
\hline
Plotly & 9 \\
Holoviews & 13 \\
Streamlit & 16 \\
Matplotlib & 22 \\
Bokeh & 29 \\
\hline
\end{tabular}
\end{table}

Najkrótszą implementację osiąga Plotly (Listing~\ref{lst:plotly_simple}),
z średnią 9 linii kodu na wykres.

\begin{lstlisting}[...]
% kod tutaj
\end{lstlisting}
```

### Przykład 4: Wyróżnienie Kluczowej Linii

```latex
\begin{lstlisting}[caption={Plotly - Grouped Bars},
                    escapeinside={(*}{*)}]
df = pd.DataFrame({...})

fig = px.bar(df, x='Operation', y='Time', 
             color='Library',
             barmode='group')  (*\colorbox{yellow}{$\leftarrow$ Kluczowy parametr!}*)
             
fig.write_html('grouped.html')
\end{lstlisting}

W linii 5 parametr \texttt{barmode='group'} automatycznie rozwiązuje 
problem pozycjonowania słupków, który w Bokeh wymaga 15+ linii 
manualnych obliczeń.
```

---

## 🎨 Customizacja Stylu

### Zmień Rozmiar Czcionki

```latex
% Mniejsza czcionka dla długiego kodu
\begin{lstlisting}[basicstyle=\ttfamily\tiny]
...
\end{lstlisting}

% Większa dla krótkiego kodu
\begin{lstlisting}[basicstyle=\ttfamily\small]
...
\end{lstlisting}
```

### Bez Numerów Linii

```latex
\begin{lstlisting}[numbers=none]
...
\end{lstlisting}
```

### Niestandardowe Kolory

```latex
\begin{lstlisting}[backgroundcolor=\color{white}, 
                    frame=none,
                    keywordstyle=\color{red}\bfseries]
...
\end{lstlisting}
```

### Bez Ramki

```latex
\begin{lstlisting}[frame=none]
...
\end{lstlisting}
```

---

## 📖 Wzorzec dla Rozdziału

### Rozdział 4.3: Porównanie Bibliotek Wizualizacyjnych

```latex
\section{Porównanie Bibliotek Wizualizacyjnych}

\subsection{Metodologia}

W celu obiektywnego porównania bibliotek, zaimplementowano 7 identycznych 
wykresów w 5 różnych frameworkach. Każda implementacja była oceniana pod 
względem:

\begin{itemize}
    \item Lines of Code (LOC) -- złożoność implementacji
    \item API Style -- deklaratywne vs imperatywne
    \item Interaktywność -- statyczne vs HTML
\end{itemize}

\subsection{Implementacja - Wykres Słupkowy}

\subsubsection{Plotly Express}

Listing~\ref{lst:plotly_bar} przedstawia implementację w Plotly Express.

\begin{lstlisting}[caption={Plotly - Simple Bar Chart},label={lst:plotly_bar}]
import plotly.express as px

df = pd.DataFrame({
    'Library': ['Pandas', 'Polars', 'PyArrow'],
    'Time': [11.0, 1.51, 5.31]
})

fig = px.bar(df, x='Library', y='Time', color='Library')
fig.write_html('output.html')
\end{lstlisting}

\textbf{Analiza:}
\begin{itemize}
    \item LOC: 8 linii
    \item Style: Deklaratywny (wszystko w jednym wywołaniu)
    \item Automatyczne tooltips i interaktywność
\end{itemize}

\subsubsection{Bokeh}

Dla porównania, Listing~\ref{lst:bokeh_bar} pokazuje implementację w Bokeh.

\begin{lstlisting}[caption={Bokeh - Simple Bar Chart},label={lst:bokeh_bar}]
from bokeh.plotting import figure, save
from bokeh.models import ColumnDataSource, HoverTool

source = ColumnDataSource(data=dict(
    libraries=['Pandas', 'Polars', 'PyArrow'],
    times=[11.0, 1.51, 5.31]
))

p = figure(x_range=['Pandas', 'Polars', 'PyArrow'],
           width=800, height=500)
p.vbar(x='libraries', top='times', width=0.7, source=source)

hover = HoverTool(tooltips=[("Library", "@libraries")])
p.add_tools(hover)

save(p, 'output.html')
\end{lstlisting}

\textbf{Analiza:}
\begin{itemize}
    \item LOC: 25 linii (212\% więcej niż Plotly)
    \item Style: Imperatywny (manualna konfiguracja)
    \item Wymaga ColumnDataSource, HoverTool
\end{itemize}

\subsection{Porównanie Ilościowe}

Tabela~\ref{tab:loc_comparison} przedstawia średnią liczbę linii kodu.

\begin{table}[h]
\centering
\caption{Porównanie LOC dla 7 wykresów}
\label{tab:loc_comparison}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Biblioteka} & \textbf{Średnia LOC} & \textbf{Wzrost vs Plotly} \\
\hline
Plotly & 9 & -- \\
Holoviews & 13 & +44\% \\
Streamlit & 16 & +78\% \\
Matplotlib & 22 & +144\% \\
Bokeh & 29 & +222\% \\
\hline
\end{tabular}
\end{table}

\subsection{Wnioski}

Na podstawie implementacji 35 wizualizacji (7 wykresów × 5 bibliotek) 
sformułowano następujące wnioski:

\begin{enumerate}
    \item \textbf{Deklaratywne API redukuje kod o 50-70\%:} Plotly i 
          Holoviews wymagają znacznie mniej kodu niż Bokeh i Matplotlib.
    
    \item \textbf{Grouped bars test complexity:} Różnice są najbardziej 
          widoczne przy złożonych układach (Listing~\ref{lst:plotly_grouped} 
          vs ~\ref{lst:bokeh_grouped}).
    
    \item \textbf{Trade-off prostota vs kontrola:} Krótszy kod (Plotly) = 
          mniej kontroli; Dłuższy kod (Bokeh) = pełna kontrola.
\end{enumerate}

\subsection{Rekomendacje}

\begin{itemize}
    \item \textbf{Dla pracy magisterskiej:} Matplotlib (PNG 300 DPI)
    \item \textbf{Dla prototypowania:} Plotly (najszybsze)
    \item \textbf{Dla appendixu:} Plotly/Holoviews (HTML interaktywny)
    \item \textbf{Dla obrony:} Streamlit (live dashboard)
\end{itemize}
```

---

## 💡 Pro Tips

### 1. **Referencje Cross-Reference**
```latex
Jak pokazano w Listing~\ref{lst:plotly_simple}...
W przeciwieństwie do Listing~\ref{lst:bokeh_simple}...
Porównaj Listing~\ref{lst:plotly_simple} z ~\ref{lst:bokeh_simple}...
```

### 2. **Inline Code**
```latex
Parametr \texttt{barmode='group'} automatycznie...
Klasa \texttt{ColumnDataSource} wymaga...
```

### 3. **Wyróżnienie Różnic**
```latex
\begin{lstlisting}[escapechar=!]
# Plotly - jedna linia
fig = px.bar(df, barmode='group')  !\colorbox{green}{✓ Automatyczne}!

# Bokeh - 15 linii
x_offset = [-0.15, 0.15, ...]  !\colorbox{red}{✗ Manualne}!
\end{lstlisting}
```

### 4. **Bibliografia**
```bibtex
@misc{plotly2024,
  title={Plotly Python Graphing Library},
  author={{Plotly Technologies Inc.}},
  year={2024},
  url={https://plotly.com/python/}
}

% W tekście:
Plotly Express \cite{plotly2024} oferuje...
```

---

## 📁 Struktura Plików

```
THESIS_COMPARISON_CHARTS/
├── CLEAN_CODE_FOR_LATEX.py        ← Python: Czyste przykłady
├── LATEX_LISTINGS_READY.tex       ← LaTeX: Gotowe listingi
├── LATEX_CODE_LISTINGS.tex        ← LaTeX: Pełny dokument
└── LATEX_INSTRUCTIONS.md          ← Ten plik
```

---

## ✅ Checklist

Przed użyciem w pracy sprawdź:

- [ ] Preambula z `listings` i `xcolor` dodana
- [ ] Styl `pythonstyle` zdefiniowany
- [ ] Labels unikalne (`lst:plotly_simple` ≠ `lst:bokeh_simple`)
- [ ] Captions opisowe
- [ ] Kod przetestowany (nie skopiowałeś z błędami)
- [ ] References działają (Listing~\ref{...})

---

## 🎓 Przykład Kompletnego Rozdziału

Zobacz plik: `LATEX_CODE_LISTINGS.tex` (linie 2000-2300)

Zawiera:
- ✅ 6 gotowych listingów
- ✅ Tabele porównawcze
- ✅ Analizę różnic
- ✅ Wnioski
- ✅ Rekomendacje

**Możesz to użyć jako template!**

---

## 🚀 Quick Test

Stwórz `test.tex`:

```latex
\documentclass{article}
\usepackage{listings}
\usepackage{xcolor}

\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstset{
    language=Python,
    backgroundcolor=\color{backcolour},
    basicstyle=\ttfamily\footnotesize,
    breaklines=true,
    numbers=left,
    frame=single
}

\begin{document}

\section{Test}

\begin{lstlisting}[caption={Plotly Test}]
import plotly.express as px
df = pd.DataFrame({'x': [1,2,3], 'y': [4,5,6]})
fig = px.bar(df, x='x', y='y')
\end{lstlisting}

\end{document}
```

Kompiluj: `pdflatex test.tex`

Jeśli działa → możesz używać w pracy! ✅

---

**Powodzenia w pisaniu pracy!** 🎓📝✨




