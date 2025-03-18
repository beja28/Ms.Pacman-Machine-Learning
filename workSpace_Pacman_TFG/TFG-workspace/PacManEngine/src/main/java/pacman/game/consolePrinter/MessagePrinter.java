package pacman.game.consolePrinter;

import java.util.List;

public class MessagePrinter {
    private boolean debugMode;

    public MessagePrinter() {
        this.debugMode = true;
    }    
    
    public MessagePrinter(boolean debugMode) {
        this.debugMode = debugMode;
    }

    public void printMessage(String mensaje, String tipo, int nivelTabulacion) {
    	
    	if (!debugMode) return;
    	
        String tabulaciones = "\t".repeat(Math.max(0, nivelTabulacion));

        String prefix;
        switch (tipo.toLowerCase()) {
            case "info":
                prefix = "[INFO] ";
                break;
            case "advertencia":
                prefix = "[ADVERTENCIA] ";
                break;
            case "error":
                prefix = "[ERROR] ";
                break;
            case "exito":
                prefix = "[ÉXITO] ";
                break;
            case "input":
                prefix = "[INPUT] ";
                break;
            case "debug":
                prefix = debugMode ? "[DEBUG] " : ""; // Solo imprime si debugMode es true
                break;
            default:
                prefix = "";
        }


        if (!prefix.isEmpty()) {
            System.out.println(tabulaciones + prefix + mensaje);
        }
    }
    
    
    public void mostrarResumenEjecucion(String fileName, int iter, int min_score, int pacManCount, int ghostCount) {
    	System.out.println();
        mostrarInfo("Comenzando a generar DataSet en archivo: '" + fileName + "'\n");
        mostrarConTabulacion("Número de iteraciones por combinación: " + iter);
        
        if (min_score == -1) {
        	mostrarConTabulacion("Score Mínimo Desactivado");
        } else {
        	mostrarConTabulacion("Score Mínimo establecido en: " + min_score);
        }
        
        mostrarConTabulacion("Modo de depuración " + (debugMode ? "activado" : "desactivado"));
        
        int combinaciones = pacManCount * ghostCount;
        int juegosTotales = combinaciones * iter;
        mostrarConTabulacion("Número de combinaciones posibles: " + combinaciones);
        mostrarConTabulacion("Número total de partidas a jugar: " + juegosTotales + "\n");
    }

    
    public void mostrarResumenFinal(long inicio, String fileName, long lineasIniciales, long lineasFinales, List<Integer> savedScores, int min_score) {
        long fin = System.nanoTime();
        long duracion = fin - inicio;
        long segundosTotales = duracion / 1_000_000_000;
        long horas = segundosTotales / 3600;
        long minutos = (segundosTotales % 3600) / 60;
        long segundos = segundosTotales % 60;
        long lineasCreadas = lineasFinales - lineasIniciales;
        
        System.out.println();
        mostrarInfo("Información de ejecución:\n");
        mostrarConTabulacion("Tiempo total: " + horas + " horas, " + minutos + " minutos, " + segundos + " segundos");
        mostrarConTabulacion("Lineas iniciales: " + lineasIniciales + ", Lineas creadas: " + lineasCreadas + ", Lineas finales: " + lineasFinales);
        
        if (!savedScores.isEmpty()) {
            double media = savedScores.stream().mapToInt(Integer::intValue).average().orElse(0.0);
            mostrarConTabulacion("Puntuación media de las partidas guardadas: " + media);
        } else {
        	mostrarConTabulacion("No se guardaron partidas con puntuaciones mayores a " + min_score);
        }
    }

    

    // Métodos específicos usando printMessage
    public void mostrarInfo(String mensaje) {
        printMessage(mensaje, "info", 0);
    }

    public void mostrarAdvertencia(String mensaje) {
        printMessage(mensaje, "advertencia", 0);
    }

    public void mostrarError(String mensaje) {
        printMessage(mensaje, "error", 0);
    }

    public void mostrarExito(String mensaje) {
        printMessage(mensaje, "exito", 0);
    }
    
    public void mostrarInput(String mensaje) {
        printMessage(mensaje, "input", 0);
    }

    public void mostrarDebug(String mensaje) {
        printMessage(mensaje, "debug", 0);
    }
    
    public void mostrarConTabulacion(String mensaje) {
    	System.out.println("\t" + mensaje);
    }
}
