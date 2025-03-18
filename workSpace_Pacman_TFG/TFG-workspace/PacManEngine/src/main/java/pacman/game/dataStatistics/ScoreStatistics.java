package pacman.game.dataStatistics;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.moment.Skewness;
import org.apache.commons.math3.stat.descriptive.moment.Kurtosis;
import pacman.game.consolePrinter.MessagePrinter;

import java.io.IOException;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.List;

public class ScoreStatistics {

    private final MessagePrinter printer;


    public ScoreStatistics(MessagePrinter printer) {
        this.printer = printer;
    }

 
    public void calcularEstadisticas(List<Integer> scores) {
        if (scores == null || scores.isEmpty()) {
            printer.mostrarError("No hay puntuaciones para calcular estadísticas.");
            return;
        }

        DescriptiveStatistics stats = new DescriptiveStatistics();
        double[] dataArray = new double[scores.size()];


        for (int i = 0; i < scores.size(); i++) {
            stats.addValue(scores.get(i));
            dataArray[i] = scores.get(i); // Para Skewness y Kurtosis
        }

        // Calcular estadísticas
        double media = stats.getMean();
        double mediana = stats.getPercentile(50);
        double desviacion = stats.getStandardDeviation();
        double varianza = stats.getVariance();
        double max = stats.getMax();
        double min = stats.getMin();
        double rango = max - min;
        double percentil25 = stats.getPercentile(25);
        double percentil75 = stats.getPercentile(75);
        double percentil90 = stats.getPercentile(90);

        // Calcular Skewness y Kurtosis
        Skewness skewnessCalc = new Skewness();
        double skewness = skewnessCalc.evaluate(dataArray);

        Kurtosis kurtosisCalc = new Kurtosis();
        double kurtosis = kurtosisCalc.evaluate(dataArray);

        printer.printMessage("------ Estadísticas Avanzadas de las partidas ------", "info", 0);
        printer.printMessage("Media: " + String.format("%.2f", media), "info", 1);
        printer.printMessage("Mediana: " + String.format("%.2f", mediana), "info", 1);
        printer.printMessage("Desviación típica: " + String.format("%.2f", desviacion), "info", 1);
        printer.printMessage("Varianza: " + String.format("%.2f", varianza), "info", 1);
        printer.printMessage("Máximo: " + String.format("%.2f", max), "info", 1);
        printer.printMessage("Mínimo: " + String.format("%.2f", min), "info", 1);
        printer.printMessage("Rango: " + String.format("%.2f", rango), "info", 1);
        printer.printMessage("Percentil 25: " + String.format("%.2f", percentil25), "info", 1);
        printer.printMessage("Percentil 75: " + String.format("%.2f", percentil75), "info", 1);
        printer.printMessage("Percentil 90: " + String.format("%.2f", percentil90), "info", 1);
        printer.printMessage("Asimetría (Skewness): " + String.format("%.2f", skewness), "info", 1);
        printer.printMessage("Curtosis (Kurtosis): " + String.format("%.2f", kurtosis) + "\n", "info", 1);
    }
    
    
    public void guardarEstadisticasEnArchivo(List<Integer> scores, String fileName) {
        if (scores == null || scores.isEmpty()) {
            printer.mostrarError("No hay puntuaciones para guardar estadísticas.");
            return;
        }

        List<String> fileContent = new ArrayList<>();
        fileContent.add("------ Estadísticas Avanzadas de las partidas ------");

        DescriptiveStatistics stats = new DescriptiveStatistics();
        double[] dataArray = new double[scores.size()];

        for (int i = 0; i < scores.size(); i++) {
            stats.addValue(scores.get(i));
            dataArray[i] = scores.get(i); // Para Skewness y Kurtosis
        }

        double media = stats.getMean();
        double mediana = stats.getPercentile(50);
        double desviacion = stats.getStandardDeviation();
        double varianza = stats.getVariance();
        double max = stats.getMax();
        double min = stats.getMin();
        double rango = max - min;
        double percentil25 = stats.getPercentile(25);
        double percentil75 = stats.getPercentile(75);
        double percentil90 = stats.getPercentile(90);

        Skewness skewnessCalc = new Skewness();
        double skewness = skewnessCalc.evaluate(dataArray);

        Kurtosis kurtosisCalc = new Kurtosis();
        double kurtosis = kurtosisCalc.evaluate(dataArray);

        fileContent.add("Media: " + String.format("%.2f", media));
        fileContent.add("Mediana: " + String.format("%.2f", mediana));
        fileContent.add("Desviación típica: " + String.format("%.2f", desviacion));
        fileContent.add("Varianza: " + String.format("%.2f", varianza));
        fileContent.add("Máximo: " + String.format("%.2f", max));
        fileContent.add("Mínimo: " + String.format("%.2f", min));
        fileContent.add("Rango: " + String.format("%.2f", rango));
        fileContent.add("Percentil 25: " + String.format("%.2f", percentil25));
        fileContent.add("Percentil 75: " + String.format("%.2f", percentil75));
        fileContent.add("Percentil 90: " + String.format("%.2f", percentil90));
        fileContent.add("Asimetría (Skewness): " + String.format("%.2f", skewness));
        fileContent.add("Curtosis (Kurtosis): " + String.format("%.2f", kurtosis));

        // Carpeta statistics
        String folderName = "statistics";
        Path folderPath = Paths.get(folderName);

        // Crear carpeta si no existe
        if (!Files.exists(folderPath)) {
            try {
                Files.createDirectories(folderPath);
            } catch (IOException e) {
                e.printStackTrace();
                return;
            }
        }

        // Crear y escribir archivo
        Path filePath = folderPath.resolve(fileName + ".txt");
        try {
            Files.write(filePath, fileContent, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
            printer.printMessage("Estadísticas guardadas en: " + filePath.toString(), "info", 0);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

