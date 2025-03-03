package pacman;

import static pacman.game.Constants.DELAY;
import static pacman.game.Constants.INTERVAL_WAIT;

import java.awt.Color;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.File;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import java.nio.file.Path;
import java.nio.file.Paths;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import pacman.controllers.Controller;
import pacman.controllers.GhostController;
import pacman.controllers.HumanController;
import pacman.controllers.PacmanController;
import pacman.game.Constants.GHOST;
import pacman.game.Constants.MOVE;
import pacman.game.Drawable;
import pacman.game.Game;
import pacman.game.GameObserver;
import pacman.game.GameView;
import pacman.game.comms.BasicMessenger;
import pacman.game.comms.Messenger;
import pacman.game.dataManager.DataSetRecorder;
import pacman.game.dataManager.GameStateFilter;
import pacman.game.internal.Node;
import pacman.game.internal.POType;
import pacman.game.util.Stats;
import pacman.game.heatmap.HeatMap;


/**
 * This class may be used to execute the game in timed or un-timed modes, with
 * or without visuals. Competitors should implement their controllers in
 * game.entries.ghosts and game.entries.pacman respectively. The skeleton
 * classes are already provided. The package structure should not be changed
 * (although you may create sub-packages in these packages).
 */
@SuppressWarnings("unused")
public class ExecutorModes {
	private final boolean pacmanPO;
	private final boolean ghostPO;
	private final boolean ghostsMessage;
	private final Messenger messenger;
	private final double scaleFactor;
	private final boolean setDaemon;
	private final boolean visuals;
	private final int tickLimit;
	private final int timeLimit;
	private final POType poType;
	private final int sightLimit;
	private final Random rnd = new Random();
	private final Function<Game, String> peek;
	private final Logger logger = LoggerFactory.getLogger(ExecutorModes.class);
	private boolean pacmanPOvisual;
	private boolean ghostsPOvisual;
	private static String VERSION = "4.4.0(ICI 24-25 stable version)";

	private static int ERROR_LOG_LEVEL = 2; // 0: no log, 1: error message, 2: error message + stack trace

	public static class Builder {
		private boolean pacmanPO = false;
		private boolean ghostPO = false;
		private boolean ghostsMessage = false;
		private Messenger messenger = new BasicMessenger();
		private double scaleFactor = 3.0d;
		private boolean setDaemon = false;
		private boolean visuals = false;
		private int tickLimit = 4000;
		private int timeLimit = 40;
		private POType poType = POType.LOS;
		private int sightLimit = 50;
		private Function<Game, String> peek = null;
		private boolean pacmanPOvisual;
		private boolean ghostsPOvisual;

		public Builder setPacmanPO(boolean po) {
			this.pacmanPO = po;
			return this;
		}

		public Builder setGhostPO(boolean po) {
			this.ghostPO = po;
			return this;
		}

		public Builder setGhostsMessage(boolean canMessage) {
			this.ghostsMessage = canMessage;
			if (canMessage) {
				messenger = new BasicMessenger();
			} else {
				messenger = null;
			}
			return this;
		}

		public Builder setMessenger(Messenger messenger) {
			this.ghostsMessage = true;
			this.messenger = messenger;
			return this;
		}

		public Builder setScaleFactor(double scaleFactor) {
			this.scaleFactor = scaleFactor;
			return this;
		}

		public Builder setGraphicsDaemon(boolean daemon) {
			this.setDaemon = daemon;
			return this;
		}

		public Builder setVisual(boolean visual) {
			this.visuals = visual;
			return this;
		}

		public Builder setTickLimit(int tickLimit) {
			this.tickLimit = tickLimit;
			return this;
		}

		public Builder setTimeLimit(int timeLimit) {
			this.timeLimit = timeLimit;
			return this;
		}

		public Builder setPOType(POType poType) {
			this.poType = poType;
			return this;
		}

		public Builder setSightLimit(int sightLimit) {
			this.sightLimit = sightLimit;
			return this;
		}

		public Builder setPeek(Function<Game, String> peek) {
			this.peek = peek;
			return this;
		}

		public ExecutorModes build() {
			System.err.println("TFG - Pacman-Machine-Learning");
			return new ExecutorModes(pacmanPO, ghostPO, ghostsMessage, messenger, scaleFactor, setDaemon,
					visuals, tickLimit, timeLimit, poType, sightLimit, peek, pacmanPOvisual, ghostsPOvisual);
		}

		public Builder setPacmanPOvisual(boolean b) {
			this.pacmanPOvisual = b;
			return this;
		}

		public Builder setGhostsPOvisual(boolean b) {
			this.ghostsPOvisual = b;
			return this;
		}
	}

	private ExecutorModes(boolean pacmanPO, boolean ghostPO, boolean ghostsMessage, Messenger messenger,
			double scaleFactor, boolean setDaemon, boolean visuals, int tickLimit, int timeLimit, POType poType,
			int sightLimit, Function<Game, String> peek, boolean pacmanPOvisual, boolean ghostsPOvisual) {
		this.pacmanPO = pacmanPO;
		this.ghostPO = ghostPO;
		this.ghostsMessage = ghostsMessage;
		this.messenger = messenger;
		this.scaleFactor = scaleFactor;
		this.setDaemon = setDaemon;
		this.visuals = visuals;
		this.tickLimit = tickLimit;
		this.timeLimit = timeLimit;
		this.poType = poType;
		this.sightLimit = sightLimit;
		this.peek = peek;
		this.pacmanPOvisual = pacmanPOvisual;
		this.ghostsPOvisual = ghostsPOvisual;
	}

	private static void writeStat(FileWriter writer, Stats stat, int i) throws IOException {
		writer.write(String.format("%s, %d, %f, %f, %f, %f, %d, %f, %f, %f, %d%n", stat.getDescription(), i,
				stat.getAverage(), stat.getSum(), stat.getSumsq(), stat.getStandardDeviation(), stat.getN(),
				stat.getMin(), stat.getMax(), stat.getStandardError(), stat.getMsTaken()));
	}


	private Game setupGame() {
		return (this.ghostsMessage) ? new Game(rnd.nextLong(), 0, messenger.copy(), poType, sightLimit)
				: new Game(rnd.nextLong(), 0, null, poType, sightLimit);
	}

	private void handlePeek(Game game) {
		if (peek != null)
			logger.info(peek.apply(game));
	}

	
	
	public void runGame(Controller<MOVE> pacManController, GhostController ghostController, int delay, boolean showJunctions) {
        Game game = setupGame();

        precompute(pacManController, ghostController);
        
        GameView gv = (visuals) ? setupGameView(pacManController, game) : null;

        GhostController ghostControllerCopy = ghostController.copy(ghostPO);
        
        //System.out.println(Arrays.toString(game.getJunctionIndices()));
        //System.out.println(game.getJunctionIndices().length);

        while (!game.gameOver()) {
            if (tickLimit != -1 && tickLimit < game.getTotalTime()) {
                break;
            }
            handlePeek(game);
            game.advanceGame(
                    pacManController.getMove(getPacmanCopy(game), System.currentTimeMillis() + timeLimit),
                    ghostControllerCopy.getMove(getGhostsCopy(game), System.currentTimeMillis() + timeLimit));
            
            
            if(showJunctions) GameView.addPoints(game, Color.RED, game.getJunctionIndices());
	        

            try {
                Thread.sleep(delay);
            } catch (Exception e) {
            }

            if (visuals) {
                gv.repaint();
            }
        }
        System.out.println(game.getScore());
        
        postcompute(pacManController, ghostController);
    }
	
	
	
	public void runGameGenerateMultiDataSet(List<PacmanController> pacManControllers, List<GhostController> ghostControllers, int iter, String fileName, boolean DEBUG, int min_score) {
	    
	    System.out.println("\n\n[INFO] Comenzando a generar DataSet en archivo: '" + fileName + "'\n");
	    System.out.printf("\tNumero de iteraciones por combinación: " + iter + "\n");
	    if(min_score==-1) {
	        System.out.printf("\tScore Minimo Desactivado \n");
	    } else {
	        System.out.printf("\tScore Minimo establecido en: " + min_score + "\n");
	    }
	    if(DEBUG) {
	        System.out.printf("\tModo de depuración activado \n\n");
	    } else {
	        System.out.printf("\tModo de depuración desactivado \n\n");
	    }
	    
	    int combinaciones = pacManControllers.size() * ghostControllers.size();
	    int juegosTotales = combinaciones * iter;
	    System.out.printf("\tNumero de combinaciones posibles: %d\n", combinaciones);
	    System.out.printf("\tNumero total de partidas a jugar: %d\n\n", juegosTotales);
	    
	    int delay = 0;
	    List<Integer> savedScores = new ArrayList<>();
	    long inicio = System.nanoTime();
	    long lineasIniciales = DataSetRecorder.contarLineas(fileName);
	    
	    
	    //Se generan todas las combinaciones posibles
	    for (Controller<MOVE> pacManController : pacManControllers) {
	        for (GhostController ghostController : ghostControllers) {
	            for (int i = 0; i < iter; i++) {
	                
	                Game game = setupGame();
	                DataSetRecorder dataRecorder = new DataSetRecorder(game);
	                precompute(pacManController, ghostController);
	                GameView gv = (visuals) ? setupGameView(pacManController, game) : null;
	                GhostController ghostControllerCopy = ghostController.copy(ghostPO);
	                
	                while (!game.gameOver()) {
	                    if (tickLimit != -1 && tickLimit < game.getTotalTime()) {
	                        break;
	                    }
	                    handlePeek(game);
	                    
	                    MOVE pacmanMove = pacManController.getMove(getPacmanCopy(game), System.currentTimeMillis() + timeLimit);
	                    dataRecorder.collectGameState(pacmanMove);
	                    
	                    game.advanceGame(pacmanMove,
	                        ghostControllerCopy.getMove(getGhostsCopy(game), System.currentTimeMillis() + timeLimit));
	                    
	                    try {
	                        Thread.sleep(delay);
	                    } catch (Exception e) {
	                    }
	                    
	                    if (visuals) {
	                        gv.repaint();
	                    }
	                }
	                
	                if (min_score == -1 || game.getScore() > min_score) {
	                    try {
	                        dataRecorder.saveDataToCsv(fileName, true);
	                        savedScores.add(game.getScore());
							if (DEBUG) {
								System.out.println("[DEBUG] " + i + ". " + pacManController.getName() + " vs "
										+ ghostController.getName() + " - Estados guardados en: " + fileName
										+ ".csv con score: " + game.getScore());
							}
	                    } catch (IOException e) {
	                        e.printStackTrace();
	                    }
	                }
	                
	                postcompute(pacManController, ghostController);
	            }
	        }
	    }
	    
	    long fin = System.nanoTime();
	    long duracion = fin - inicio;
	    long segundosTotales = duracion / 1_000_000_000;
	    long horas = segundosTotales / 3600;
	    long minutos = (segundosTotales % 3600) / 60;
	    long segundos = segundosTotales % 60;   
	    long lineasFinales = DataSetRecorder.contarLineas(fileName);
	    long lineasCreadas = lineasFinales - lineasIniciales;
	    
	    System.out.println("\n\n[INFO] Información de ejecución:\n");
	    System.out.printf("\tTiempo total: %d horas, %d minutos, %d segundos%n", horas, minutos, segundos);
	    System.out.println("\tLineas iniciales: " + lineasIniciales + ", Lineas creadas: " + lineasCreadas + ", Lineas finales: " + lineasFinales);
	    
	    if (!savedScores.isEmpty()) {
	        double media = savedScores.stream().mapToInt(Integer::intValue).average().orElse(0.0);
	        System.out.println("\tPuntuación media de las partidas guardadas: " + media);
	    } else {
	        System.out.println("\tNo se guardaron partidas con puntuaciones mayores a " + min_score);
	    }
	}
	
	
	
	public void runGameHeatMaps(Controller<MOVE> pacManController, GhostController ghostController, int delay) {
		boolean imageSaved = false;
		
		String directorioActual = System.getProperty("user.dir"); // Esto te da el directorio raíz del proyecto
		
		// Construir la ruta relativa a 'mapas_explicabilidad' (Ajustar en funcion de si es sklearn o pytorch)
		Path mapsPath = Paths.get(directorioActual, "mapas_explicabilidad_txt_pytorch"); 
		
	    Game game = setupGame();
	    precompute(pacManController, ghostController);

	    GameView gv = (visuals) ? setupGameView(pacManController, game) : null;
	    GhostController ghostControllerCopy = ghostController.copy(ghostPO);

	    // Lista de las 10 características mas importantes segun la explicabilidad en sklearn
	    String[] selectedFeatures_sklearn = {
	        "score", "euclideanDistanceToPp", "pathDistanceToPp", "totalTime", "timeOfLastGlobalReversal",
	        "ghost3NodeIndex", "pacmanLastMoveMade_LEFT", "pacmanLastMoveMade_UP", "pacmanLastMoveMade_DOWN", "ghost1NodeIndex"
	    };
	    
	    // Lista de las 10 características mas importantes segun la explicabilidad en pytorch
	    String[] selectedFeatures_pytorch = {
	        "score", "totalTime", "timeOfLastGlobalReversal", "ghost3NodeIndex", "ghost4NodeIndex",
	        "ghost2NodeIndex", "ghost1NodeIndex", "scoreDiff50", "scoreDiff25", "ghost3EdibleTime"
	    };

	    // Carpeta donde se guardarán los mapas de calor
	    Path heatmapFolder = Paths.get(directorioActual, "mapas_pytorch");
	    new File(heatmapFolder.toString()).mkdirs(); // Crea la carpeta si no existe

	    Map<String, Map<Integer, Double>> heatmapData = new HashMap<>();

	    // Cargar los datos de los 10 archivos seleccionados (elegir entre sklearn / pytorch)
	    for (String feature : selectedFeatures_pytorch) {
	        String filePath = mapsPath.toString();
	        Map<Integer, Double> featureData = HeatMap.loadHeatMapData(filePath, feature);
	        heatmapData.put(feature, featureData);
	    }
	    

	    while (!game.gameOver()) {
	        if (tickLimit != -1 && tickLimit < game.getTotalTime()) {
	            break;
	        }

	        handlePeek(game);
	        game.advanceGame(
	            pacManController.getMove(getPacmanCopy(game), System.currentTimeMillis() + timeLimit),
	            ghostControllerCopy.getMove(getGhostsCopy(game), System.currentTimeMillis() + timeLimit)
	        );

	        // Para cada característica, dibuja su mapa de calor (cambiar dependiendo de sklearn o pytorch)
	        for (String feature : selectedFeatures_pytorch) {
	            Map<Integer, Double> data = heatmapData.get(feature);
	            if (data != null) {
	                for (Map.Entry<Integer, Double> entry : data.entrySet()) {
	                    int intersection = entry.getKey();
	                    double impact = entry.getValue();
	                    Color color = HeatMap.getColorFromImpact(impact);
	                    GameView.addPoints(game, color, intersection);
	                }
	            }
	            
		        if(!imageSaved && visuals) {
		        	HeatMap.saveMap(game, heatmapFolder, gv, feature); // Guardar el mapa solo en el primer tick
		        }
	            
	        }
	        
        	imageSaved = true;
	        
	        

	        try {
	            Thread.sleep(delay);
	        } catch (Exception e) {
	            e.printStackTrace();
	        }

	        if (visuals) {
	            gv.repaint();
	        }
	    }


	    postcompute(pacManController, ghostController);
	}	
	

	private void postcompute(Controller<MOVE> pacManController, GhostController ghostController) {
		pacManController.postCompute();
		ghostController.postCompute();
	}

	private void precompute(Controller<MOVE> pacManController, GhostController ghostController) {
		String ghostName = ghostController.getClass().getCanonicalName();
		String pacManName = pacManController.getClass().getCanonicalName();
		//System.out.print("Precompute " + ghostName);
		pacManController.preCompute(ghostName);
		//System.out.println(" ...done");
		//System.out.print("Precompute " + pacManName);
		ghostController.preCompute(pacManName);
		//System.out.println(" ...done");

	}

	private Game getPacmanCopy(Game game) {
		return game.copy((pacmanPO) ? Game.PACMAN : Game.CLONE);
	}

	private Game getGhostsCopy(Game game) {
		return game.copy((ghostPO) ? Game.ANY_GHOST : Game.CLONE);

	}

	private GameView setupGameView(Controller<MOVE> pacManController, Game game) {
		GameView gv;
		gv = new GameView(game, setDaemon);
		gv.setScaleFactor(scaleFactor);
		gv.showGame("");
		if (pacmanPOvisual)
			gv.setPacManPO(this.pacmanPO);
		if (ghostsPOvisual)
			gv.setGhostPO(this.ghostPO);
		if (pacManController instanceof HumanController) {
			gv.setFocusable(true);
			gv.requestFocus();
			gv.setPacManPO(this.pacmanPO);
			gv.addKeyListener(((HumanController) pacManController).getKeyboardInput());
		}

		if (pacManController instanceof Drawable) {
			gv.addDrawable((Drawable) pacManController);
		}
		return gv;
	}

	private GameView setupGameView(Controller<MOVE> pacManController, Game game, String title) {
		GameView gv;
		gv = new GameView(game, setDaemon);
		gv.setScaleFactor(scaleFactor);
		gv.showGame(title);
		if (pacmanPOvisual)
			gv.setPacManPO(this.pacmanPO);
		if (ghostsPOvisual)
			gv.setGhostPO(this.ghostPO);
		if (pacManController instanceof HumanController) {
			gv.setFocusable(true);
			gv.requestFocus();
			gv.setPacManPO(this.pacmanPO);
			gv.addKeyListener(((HumanController) pacManController).getKeyboardInput());
		}

		if (pacManController instanceof Drawable) {
			gv.addDrawable((Drawable) pacManController);
		}
		return gv;
	}


	public static void logError(String msg, Exception e) {
		if (ExecutorModes.ERROR_LOG_LEVEL < 0)
			System.err.println(msg);
		if (ExecutorModes.ERROR_LOG_LEVEL < 1)
			e.printStackTrace(System.err);

	}
}