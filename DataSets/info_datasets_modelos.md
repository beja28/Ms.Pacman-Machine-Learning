# Documentación de Modelos Entrenados

| Nombre del Modelo | Tipo de Modelo | DataSet Utilizado     | Experimento | Estadísticas de las Partidas                                                            |
|-------------------|----------------|-----------------------|------------|-----------------------------------------------------------------------------------------|
| models_2024-12-13 | Sklearn        | N/A                   | Fallido    | -                                                                                       |
| models_2024-12-14 | Pytorch        | N/A                   | Fallido    | -                                                                                       |
| models_2025-02-12 | Sklearn        | N/A                   | Fallido    | -                                                                                       |
| models_2025-02-14 | Pytorch        | N/A                   | Fallido    | -                                                                                       |
| models_2025-03-05 | Pytorch        | 06_gameStatesData.csv | 2          | [Ver detalles](#estadisticas-modelo-2025-03-05 "Ir a sección detallada de estadísticas")|
| models_2025-03-12 | Sklearn        | 06_gameStatesData.csv | 2          | [Ver detalles](#estadisticas-modelo-2025-03-12 "Ir a sección detallada de estadísticas")|
| models_2025-03-29 | TabNet     | 18_gameStatesData_enriched.csv           | 3          | [Ver detalles](#estadisticas-modelo-2025-03-29 "Ir a sección detallada de estadísticas")                          |
| models_2025-04-26 | TabNet     | 23_gameStatesData_enriched.csv           | 4          | [Ver detalles](#estadisticas-modelo-2025-04-26 "Ir a sección detallada de estadísticas")                          |

---

## Estadisticas-Modelo-2025-03-05
**Rivales (100 partidas por cada uno)**  
- es.ucm.fdi.ici.c2324.practica1.grupo06.Ghosts()  
- es.ucm.fdi.ici.c2324.practica1.grupo01.Ghosts()  
- es.ucm.fdi.ici.c2324.practica1.grupo08.Ghosts()  
- es.ucm.fdi.ici.c2324.practica2.grupo04.Ghosts()  
- es.ucm.fdi.ici.c2324.practica2.grupo01.Ghosts()  
- es.ucm.fdi.ici.c2324.practica2.grupo02.Ghosts()

------

**Estadísticas Avanzadas**  
- Media: 2368,30  
- Mediana: 2250,00  
- Desviación típica: 1119,38  
- Varianza: 1253008,79  
- Máximo: 7470,00  
- Mínimo: 330,00  
- Rango: 7140,00  
- Percentil 25: 1560,00  
- Percentil 75: 2940,00  
- Percentil 90: 3839,00  
- Asimetría (Skewness): 0,92  
- Curtosis (Kurtosis): 1,27

![Resultado](images\pytorch_models_2025-03-05_histograma.png)
![Resultado](images\pytorch_models_2025-03-05_boxplot.png)


## Estadisticas-Modelo-2025-03-12
**Rivales (100 partidas por cada uno)**  
- es.ucm.fdi.ici.c2324.practica1.grupo06.Ghosts()  
- es.ucm.fdi.ici.c2324.practica1.grupo01.Ghosts()  
- es.ucm.fdi.ici.c2324.practica1.grupo08.Ghosts()  
- es.ucm.fdi.ici.c2324.practica2.grupo04.Ghosts()  
- es.ucm.fdi.ici.c2324.practica2.grupo01.Ghosts()  
- es.ucm.fdi.ici.c2324.practica2.grupo02.Ghosts()

------

**Estadísticas Avanzadas**  
- Media: 2101,43  
- Mediana: 2055,00
- Desviación típica: 878,35
- Varianza: 771498,11 
- Máximo: 7490,00 
- Mínimo: 350,00  
- Rango: 7140,00  
- Percentil 25: 1452,50  
- Percentil 75: 2570,00 
- Percentil 90: 3089,00  
- Asimetría (Skewness): 1,00
- Curtosis (Kurtosis): 2,75

![Resultado](images\sklearn_models_2025-03-12_histograma.png)
![Resultado](images\sklearn_models_2025-03-12_boxplot.png)



## Estadisticas-Modelo-2025-03-29
**Rivales (100 partidas por cada uno)**  
- es.ucm.fdi.ici.c2324.practica1.grupo06.Ghosts()  
- es.ucm.fdi.ici.c2324.practica1.grupo01.Ghosts()  
- es.ucm.fdi.ici.c2324.practica1.grupo08.Ghosts()  
- es.ucm.fdi.ici.c2324.practica2.grupo04.Ghosts()  
- es.ucm.fdi.ici.c2324.practica2.grupo01.Ghosts()  
- es.ucm.fdi.ici.c2324.practica2.grupo02.Ghosts()

------

**Estadísticas Avanzadas**  
- Media: 2479,68
- Mediana: 2390,00
- Desviación típica: 814,57
- Varianza: 663523,27
- Máximo: 6550,00
- Mínimo: 320,00
- Rango: 6230,00
- Percentil 25: 1990,00
- Percentil 75: 2890,00
- Percentil 90: 3509,00
- Asimetría (Skewness): 0,95
- Curtosis (Kurtosis): 2,70

![Resultado](images\tabnet_models_2025-03-29_histograma.png)
![Resultado](images\tabnet_models_2025-03-29_boxplot.png)



## Estadisticas-Modelo-2025-04-26
**Rivales (500 partidas contra la mejor implementacion de fantasmas (3704))**  
es.ucm.fdi.ici.c2223.practica2.grupo02.Ghosts 

------

**Estadísticas Avanzadas**  
- Media: 2630,08
- Mediana: 2510,00
- Desviación típica: 877,15
- Varianza: 769390,22
- Máximo: 5880,00
- Mínimo: 590,00
- Rango: 5290,00
- Percentil 25: 2110,00
- Percentil 75: 3117,50
- Percentil 90: 3669,00
- Asimetría (Skewness): 0,74
- Curtosis (Kurtosis): 1,56

![Resultado](images\histograma_tabnet_Exp4.jpg)
![Resultado](images\boxplot_tabnet_Exp4.jpg)
