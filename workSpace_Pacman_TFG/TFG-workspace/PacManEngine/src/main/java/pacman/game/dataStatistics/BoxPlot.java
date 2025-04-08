package pacman.game.dataStatistics;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.renderer.category.BoxAndWhiskerRenderer;
import org.jfree.data.statistics.DefaultBoxAndWhiskerCategoryDataset;

import pacman.game.consolePrinter.MessagePrinter;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

public class BoxPlot {

    private static JFreeChart buildBoxPlotChart(List<Integer> scores) {
        DefaultBoxAndWhiskerCategoryDataset dataset = new DefaultBoxAndWhiskerCategoryDataset();
        List<Double> scoresAsDouble = scores.stream()
                .map(Integer::doubleValue)
                .collect(Collectors.toList());

        dataset.add(scoresAsDouble, "", "");

        JFreeChart chart = ChartFactory.createBoxAndWhiskerChart(
                "Distribución de Puntuaciones",
                "Distribución",
                "Puntuaciones",
                dataset,
                false
        );

        chart.setBackgroundPaint(Color.WHITE);

        CategoryPlot plot = (CategoryPlot) chart.getPlot();
        plot.setOrientation(org.jfree.chart.plot.PlotOrientation.HORIZONTAL);
        plot.setBackgroundPaint(Color.WHITE);
        plot.setDomainGridlinePaint(Color.GRAY);
        plot.setRangeGridlinePaint(Color.GRAY);

        BoxAndWhiskerRenderer renderer = (BoxAndWhiskerRenderer) plot.getRenderer();
        renderer.setSeriesOutlinePaint(0, Color.BLACK);
        renderer.setSeriesPaint(0, new Color(0, 102, 204, 200)); // Azul con opacidad
        renderer.setArtifactPaint(Color.RED); // Outliers
        renderer.setMaximumBarWidth(0.30);
        renderer.setSeriesOutlineStroke(0, new BasicStroke(2.5f));
        

        plot.getDomainAxis().setLabelFont(new Font("SansSerif", Font.BOLD, 14));
        plot.getRangeAxis().setLabelFont(new Font("SansSerif", Font.BOLD, 14));
        chart.getTitle().setFont(new Font("SansSerif", Font.BOLD, 16));

        return chart;
    }

    // Lo crea y lo muestra
    public static void createBoxPlot(List<Integer> scores) {
        JFreeChart chart = buildBoxPlotChart(scores);

        JFrame frame = new JFrame("Boxplot");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.pack();
        frame.setVisible(true);
    }

    // Lo guarda en la carpeta de estadisticas
    public static void guardarGraficaEnArchivo(List<Integer> scores, String fileName) {
        JFreeChart chart = buildBoxPlotChart(scores);

        MessagePrinter printer = new MessagePrinter();
        Path outputPath = Paths.get("statistics", "boxplots", fileName + "_boxplot.png");
        try {
            Files.createDirectories(outputPath.getParent());
            ChartUtils.saveChartAsPNG(outputPath.toFile(), chart, 800, 600);
            printer.printMessage("Archivo .png del box-plot guardado en: " + outputPath.toString(), "info", 0);
        } catch (IOException e) {
        	printer.mostrarError("Error al guardar el archivo .png del box-pot.");
        }
    }
}
