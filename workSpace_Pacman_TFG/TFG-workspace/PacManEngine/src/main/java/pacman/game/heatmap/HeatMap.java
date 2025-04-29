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

	public static Color getColorFromImpact(double impact, String model) {
	    double absImpact = Math.abs(impact); // Trabajamos con el valor absoluto para la escala de color
	    double ratio = 0;

	    // Definir límites para la interpolación de colores
	    double maxImpact = 0;  // Impacto máximo
	    
	    if(model == "pytorch") {
		    maxImpact = 30;
		    ratio = Math.min(1, absImpact / maxImpact); // Escalar el impacto dentro del rango

	    }
	    else if(model == "sklearn") {
	        maxImpact = 0.7;
	        
	    	 // Escalamos el impacto en relación a su distancia con 0 (máximo impacto en 1 o -1)
	        ratio = Math.min(1, absImpact / maxImpact); // Como los valores pocas veces superan 1, usamos 1 como referencia
	    }
	    else if(model == "tabnet") {
	        maxImpact = 0.024;
	        
	        ratio = Math.min(1, absImpact / maxImpact); // Como los valores nunca superan 1, usamos 1 como referencia
	    }


	    // Transición de colores: Rojo -> Naranja -> Amarillo -> Verde
	    int red, green;

	    if (ratio < 0.33) {
	        // Rojo -> Naranja
	        red = 255;
	        green = (int) (ratio * 3 * 128);  
	    } else if (ratio < 0.66) {
	        // Naranja -> Amarillo
	        red = 255;
	        green = (int) (128 + (ratio - 0.33) * 3 * 127);
	    } else {
	        // Amarillo -> Verde
	        red = (int) (255 - (ratio - 0.66) * 3 * 255);
	        green = 255;
	    }

	    // Asegurar que los valores de los colores no superen 255
	    red = Math.min(255, Math.max(0, red));
	    green = Math.min(255, Math.max(0, green));

	    return new Color(red, green, 0);
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
