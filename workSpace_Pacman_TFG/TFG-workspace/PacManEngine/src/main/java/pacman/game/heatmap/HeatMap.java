package pacman.game.heatmap;
import java.awt.Color;
import java.awt.Dimension;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import java.nio.file.Path;
import java.nio.file.Paths;

import java.awt.Graphics2D;  
import java.awt.image.BufferedImage;  

import javax.imageio.ImageIO;

import pacman.game.Game;
import pacman.game.GameView;

import java.util.HashMap;
import java.util.Map;

public class HeatMap {

	public static Color getColorFromImpact(double impact) {
        // Normalizar el impacto en el rango [-1, 1] para que los colores sean más consistentes
        double normalizedImpact = Math.max(-1, Math.min(1, impact));

        int red, green;
        
        if (normalizedImpact < 0) {  
            // Valores negativos: de amarillo a rojo
            red = 255;
            green = (int) (255 * (1 + normalizedImpact)); // Se reduce el verde a medida que se acerca a -1
        } else {  
            // Valores positivos: de amarillo a verde
            green = 255;
            red = (int) (255 * (1 - normalizedImpact)); // Se reduce el rojo a medida que se acerca a 1
        }

        return new Color(red, green, 0); // Azul siempre 0 para mantener el espectro rojo-amarillo-verde
    }

    // Método para leer el archivo y devolver un mapa con intersecciones y sus impactos
    public static Map<Integer, Double> loadHeatMapData(String filePath, String featureName) {
        Map<Integer, Double> heatMapData = new HashMap<>();
        
        // Crear la ruta completa hacia el archivo de la característica
        Path featureFilePath = Paths.get(filePath, featureName + ".txt");
        
        try (BufferedReader br = new BufferedReader(new FileReader(featureFilePath.toString()))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.startsWith("Intersecci�n")) { // Se pone asi porque en el txt esta guardado con tilde
                    String[] parts = line.split(":");
                    if (parts.length == 2) {
                        // Extraer el índice de la intersección y el valor del impacto
                        int intersection = Integer.parseInt(parts[0].replaceAll("\\D+", ""));
                        double impact = Double.parseDouble(parts[1].trim());
                        heatMapData.put(intersection, impact);
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        return heatMapData;
    }
    
    public static void saveMap(Game game, Path heatmapFolder, GameView gv, String feature) {
    	Dimension preferredSize = gv.getPreferredSize();
    	// Crea la imagen para el tablero actual
        BufferedImage bufferedImage = new BufferedImage(preferredSize.width, preferredSize.height, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = bufferedImage.createGraphics();
        gv.paint(g2d); // Usamos el paint() para obtener la imagen del tablero

        // Guardamos la imagen
        try {
            Path outputPath = heatmapFolder.resolve("heatmap_" + feature + ".png");
            ImageIO.write(bufferedImage, "PNG", outputPath.toFile());
            System.out.println("Imagen guardada: " + outputPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
