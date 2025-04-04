package pacman.game.dataStatistics;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYBarRenderer;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.data.statistics.HistogramType;

import pacman.game.consolePrinter.MessagePrinter;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class Histogram {

    public static void createHistogram(List<Integer> scores) {
        double[] values = scores.stream().mapToDouble(Integer::doubleValue).toArray();

        HistogramDataset dataset = new HistogramDataset();
        dataset.setType(HistogramType.FREQUENCY);
        dataset.addSeries("Puntuaciones", values, 10);

        JFreeChart histogram = ChartFactory.createHistogram(
                "Distribución de Puntuaciones",
                "Puntuación",
                "Frecuencia",
                dataset,
                PlotOrientation.VERTICAL,
                false,
                true,
                false
        );

        XYPlot plot = (XYPlot) histogram.getPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setRangeGridlinePaint(Color.GRAY);

        XYBarRenderer renderer = (XYBarRenderer) plot.getRenderer();
        renderer.setSeriesPaint(0, new Color(0, 102, 204));
        renderer.setDrawBarOutline(true);
        renderer.setSeriesOutlinePaint(1, Color.BLACK);
        renderer.setSeriesOutlineStroke(1, new BasicStroke(2.0f));
        renderer.setBarPainter(new org.jfree.chart.renderer.xy.StandardXYBarPainter());
        renderer.setMargin(0.05);
        renderer.setShadowVisible(false);

        histogram.getTitle().setFont(new Font("SansSerif", Font.BOLD, 16));
        plot.getDomainAxis().setLabelFont(new Font("SansSerif", Font.BOLD, 14));
        plot.getRangeAxis().setLabelFont(new Font("SansSerif", Font.BOLD, 14));

        JFrame frame = new JFrame("Histograma");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(histogram));
        frame.pack();
        frame.setVisible(true);
    }

    
    //Guarda el archivo .png en la carpeta de statistics
    public static void guardarGraficaEnArchivo(List<Integer> scores, String fileName) {
        double[] values = scores.stream().mapToDouble(Integer::doubleValue).toArray();

        HistogramDataset dataset = new HistogramDataset();
        dataset.setType(HistogramType.FREQUENCY);
        dataset.addSeries("Puntuaciones", values, 10);

        JFreeChart histogram = ChartFactory.createHistogram(
                "Distribución de Puntuaciones",
                "Puntuación",
                "Frecuencia",
                dataset,
                PlotOrientation.VERTICAL,
                false,
                true,
                false
        );

        XYPlot plot = (XYPlot) histogram.getPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setRangeGridlinePaint(Color.GRAY);

        XYBarRenderer renderer = (XYBarRenderer) plot.getRenderer();
        renderer.setSeriesPaint(0, new Color(0, 102, 204));
        renderer.setDrawBarOutline(true);
        renderer.setSeriesOutlinePaint(1, Color.BLACK);
        renderer.setSeriesOutlineStroke(1, new BasicStroke(2.0f));
        renderer.setBarPainter(new org.jfree.chart.renderer.xy.StandardXYBarPainter());
        renderer.setMargin(0.05);
        renderer.setShadowVisible(false);

        histogram.getTitle().setFont(new Font("SansSerif", Font.BOLD, 16));
        plot.getDomainAxis().setLabelFont(new Font("SansSerif", Font.BOLD, 14));
        plot.getRangeAxis().setLabelFont(new Font("SansSerif", Font.BOLD, 14));

        MessagePrinter printer = new MessagePrinter();
        try {        	
            Path outputPath = Paths.get("statistics", "histograms", fileName + "_histograma.png");
            Files.createDirectories(outputPath.getParent());
            ChartUtils.saveChartAsPNG(outputPath.toFile(), histogram, 800, 600);
            printer.printMessage("Archivo .png del histograma guardado en: " + outputPath.toString(), "info", 0);
        } catch (IOException e) {
            printer.mostrarError("Error al guardar el archivo .png del histograma.");
        }
    }
}
