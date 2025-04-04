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
import pacman.game.consolePrinter.MessagePrinter;
import pacman.game.consolePrinter.UserPrompt;
import pacman.game.dataManager.DataSetRecorder;
import pacman.game.dataManager.GameStateFilter;
import pacman.game.dataStatistics.BoxPlot;
import pacman.game.dataStatistics.Histogram;
import pacman.game.dataStatistics.ScoreStatistics;
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
            
            /* Por si se quiere saber el numero de interseccion por el cual esta pasando en ese instante
            int[] intersecciones = game.getJunctionIndices();
            for(int i: intersecciones) {
            	if(i == game.getPacmanCurrentNodeIndex()) {
                	System.out.println(game.getPacmanCurrentNodeIndex());
                }
            }
            */
            
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
	
	
	
	public void runGameCalculateAverageScore(PacmanController pacManController, List<GhostController> ghostControllers,
			int iter, int delay, String fileName) {

		MessagePrinter printer = new MessagePrinter();
		List<Integer> scores = new ArrayList<>();

		for (GhostController ghostController : ghostControllers) {
			for (int i = 0; i < iter; i++) {

				Game game = setupGame();
				precompute(pacManController, ghostController);
				GhostController ghostControllerCopy = ghostController.copy(ghostPO);

				printer.mostrarInfo("Comenzando partida entre " + pacManController.getName() + " vs "
						+ ghostController.getName());
				
				
				try {
					while (!game.gameOver()) {
						if (tickLimit != -1 && tickLimit < game.getTotalTime()) {
							break;
						}

						game.advanceGame(
								pacManController.getMove(getPacmanCopy(game), System.currentTimeMillis() + timeLimit),
								ghostControllerCopy.getMove(getGhostsCopy(game),
										System.currentTimeMillis() + timeLimit));
					}

					scores.add(game.getScore());

				} catch (Exception e) {
					printer.mostrarError("Se produjo un error durante la ejecucion del juego entre: "
							+ pacManController.getName() + " vs " + ghostController.getName() + "\n");
				} finally {
					postcompute(pacManController, ghostController);

					try {
						printer.mostrarInfo("Partida entre " + pacManController.getName() + " vs "
								+ ghostController.getName() + " finalizada con éxito y score: " + game.getScore() + "\n");
						Thread.sleep(delay);
					} catch (Exception e) {}
				}
			}
		}
		
		// Mostrar estadisticas
		ScoreStatistics scoreStats = new ScoreStatistics(printer);
		scoreStats.calcularEstadisticas(scores);
		
		// Mostrar histograma
		Histogram.createHistogram(scores);
		
		// Mostrar BoxPlot
		BoxPlot.createBoxPlot(scores);
		
		if(fileName != "") {
			scoreStats.guardarEstadisticasEnArchivo(scores, fileName);
			Histogram.guardarGraficaEnArchivo(scores, fileName);
			BoxPlot.guardarGraficaEnArchivo(scores, fileName);
		}
	}
	
	public void runGameGenerateMultiDataSet(List<PacmanController> pacManControllers, List<GhostController> ghostControllers, int iter, String fileName, boolean DEBUG, int min_score) {
	    
		//Mensajes de informacion
	    MessagePrinter printer = new MessagePrinter(true);
	    MessagePrinter debug_printer= new MessagePrinter(DEBUG);
	    printer.mostrarResumenEjecucion(fileName, iter, min_score, pacManControllers.size(), ghostControllers.size());
	    
	    //Confirmacion para comenzar ejecucion
	    UserPrompt userPrompt = new UserPrompt(printer);
	    if (!UserPrompt.solicitarConfirmacionEjecucion()) {
	        return;
	    }
	    
	    List<Integer> savedScores = new ArrayList<>();
	    long inicio = System.nanoTime();
	    long lineasIniciales = DataSetRecorder.contarLineas(fileName);
	    int partidasJugadas = 0;
	    int porcentajeMostrado = 0;
	    int juegosTotales = pacManControllers.size() * ghostControllers.size() * iter;
	    
	    for (Controller<MOVE> pacManController : pacManControllers) {
	        for (GhostController ghostController : ghostControllers) {
	            for (int i = 0; i < iter; i++) {
	                
	                Game game = setupGame();
	                DataSetRecorder dataRecorder = new DataSetRecorder(game);
	                precompute(pacManController, ghostController);
	                GameView gv = (visuals) ? setupGameView(pacManController, game) : null;
	                GhostController ghostControllerCopy = ghostController.copy(ghostPO);
	                boolean gameSuccess = true;
	                
	                try {
	                    while (!game.gameOver()) {
	                        if (tickLimit != -1 && tickLimit < game.getTotalTime()) {
	                            break;
	                        }
	                        handlePeek(game);
	                        
	                        MOVE pacmanMove = pacManController.getMove(getPacmanCopy(game), System.currentTimeMillis() + timeLimit);
	                        dataRecorder.collectGameState(pacmanMove, game);
	                        
	                        game.advanceGame(pacmanMove,
	                            ghostControllerCopy.getMove(getGhostsCopy(game), System.currentTimeMillis() + timeLimit));
	                       
	                        if (visuals) {
	                            gv.repaint();
	                        }
	                    }
	                } catch (Exception e) {
	                    printer.mostrarError("Se produjo un error durante la ejecucion del juego entre: " + pacManController.getName() + " vs "
                                + ghostController.getName());
	                    gameSuccess = false;
	                }
	                
	                if (gameSuccess && (min_score == -1 || game.getScore() > min_score)) {
	                    try {
	                        dataRecorder.saveDataToCsv(fileName, true);
	                        savedScores.add(game.getScore());
	                        if (DEBUG) {
	                            debug_printer.mostrarDebug((partidasJugadas+1) + ". " + pacManController.getName() + " vs " + ghostController.getName() + " - Estados guardados en: " + fileName + ".csv con score: " + game.getScore());
	                        }
	                    } catch (IOException e) {
	                        e.printStackTrace();
	                    }
	                }
	                
	                postcompute(pacManController, ghostController);
	                
	                partidasJugadas++;
	                int nuevoPorcentaje = (partidasJugadas * 100) / juegosTotales;
	                if (nuevoPorcentaje >= porcentajeMostrado + 1) {
	                    porcentajeMostrado = nuevoPorcentaje;
	                    printer.mostrarInfo("Progreso: " + porcentajeMostrado + "% completado (" + partidasJugadas + "/" + juegosTotales + " ejecuciones)");
	                }
	            }
	        }
	    }
	    
	    //Mostrar resumen final de ejecucion
	    printer.mostrarResumenFinal(inicio, fileName, lineasIniciales, DataSetRecorder.contarLineas(fileName), savedScores, min_score);
	    
	    // Mostrar estadisticas
	 	ScoreStatistics scoreStats = new ScoreStatistics(printer);
	 	scoreStats.calcularEstadisticas(savedScores);
	 	
	 	// Mostrar histograma
	 	Histogram.createHistogram(savedScores);
	 		
	 	// Mostrar BoxPlot
	 	BoxPlot.createBoxPlot(savedScores);
	 		
	 	if(fileName != "") {
	 		scoreStats.guardarEstadisticasEnArchivo(savedScores, fileName + "_statistics");
	 		Histogram.guardarGraficaEnArchivo(savedScores, fileName);
			BoxPlot.guardarGraficaEnArchivo(savedScores, fileName);
	 	}
	 	
	 	//Mostrar el numero de movimientos invalidos
	 	printer.mostrarInfo("Movimientos inválidos en intersecciones " + DataSetRecorder.getInvalidMoveRatio());
	}

	
	
	
	public void runGameHeatMaps(Controller<MOVE> pacManController, GhostController ghostController, int delay, String model) {
		boolean imageSaved = false;
		String[] model_features = null;
		String name_save_dir = null;
		String name_load_dir = null;
		
	    // Lista de las 10 características mas importantes segun la explicabilidad en sklearn
	    String[] selectedFeatures_sklearn = {
	        "score", "totalTime", "timeOfLastGlobalReversal", "remainingPp", "ghost1NodeIndex",
	        "euclideanDistanceToPp", "ghost3NodeIndex", "pathDistanceToPp", "ghost4NodeIndex", "ghost1Distance"
	    };
	    
	    // Lista de las 10 características mas importantes segun la explicabilidad en pytorch
	    String[] selectedFeatures_pytorch = {
	        "score", "totalTime", "timeOfLastGlobalReversal", "ghost3NodeIndex", "ghost4NodeIndex",
	        "ghost2NodeIndex", "ghost1NodeIndex", "scoreDiff50", "scoreDiff25", "ghost3EdibleTime"
	    };
		
		if(model == "pytorch") {
			model_features = selectedFeatures_pytorch;
			name_save_dir = "mapas_pytorch";
			name_load_dir = "mapas_explicabilidad_txt_pytorch";
		}
		else if(model == "sklearn") {
			model_features = selectedFeatures_sklearn;
			name_save_dir = "mapas_sklearn";
			name_load_dir = "mapas_explicabilidad_txt_sklearn";
		}
		
		String directorioActual = System.getProperty("user.dir"); // Esto te da el directorio raíz del proyecto
		
		// Construir la ruta relativa a 'mapas_explicabilidad' (Ajustar en funcion de si es sklearn o pytorch)
		Path mapsPath = Paths.get(directorioActual, name_load_dir); 
		
	    Game game = setupGame();
	    precompute(pacManController, ghostController);

	    GameView gv = (visuals) ? setupGameView(pacManController, game) : null;
	    GhostController ghostControllerCopy = ghostController.copy(ghostPO);


	    // Carpeta donde se guardarán los mapas de calor
	    Path heatmapFolder = Paths.get(directorioActual, name_save_dir);
	    new File(heatmapFolder.toString()).mkdirs(); // Crea la carpeta si no existe

	    Map<String, Map<Integer, Double>> heatmapData = new HashMap<>();

	    // Cargar los datos de los 10 archivos seleccionados (elegir entre sklearn / pytorch)
	    for (String feature : model_features) {
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
	        for (String feature : model_features) {
	            Map<Integer, Double> data = heatmapData.get(feature);
	            if (data != null) {
	                for (Map.Entry<Integer, Double> entry : data.entrySet()) {
	                    int intersection = entry.getKey();
	                    double impact = entry.getValue();
	                    Color color = HeatMap.getColorFromImpact(impact, model);
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