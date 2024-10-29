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
import java.net.Socket;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import pacman.controllers.Controller;
import pacman.controllers.GhostController;
import pacman.controllers.HumanController;
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
			System.err.println("MsPacMan Engine - Ingeniería de Comportamientos Inteligentes. Version "
					+ ExecutorModes.VERSION);
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

	
	
	public int runGame(Controller<MOVE> pacManController, GhostController ghostController, int delay) {
        Game game = setupGame();

        precompute(pacManController, ghostController);
        
        GameView gv = (visuals) ? setupGameView(pacManController, game) : null;

        GhostController ghostControllerCopy = ghostController.copy(ghostPO);
        
        
        // Lista con las distintas posiciones de los estados guardados en el DataSet
        List<Integer> intersecciones = Arrays.asList(151, 153, 163, 165, 177, 188, 189, 201, 213, 225, 237, 308, 348, 361, 386, 399, 430, 456, 475, 480, 516, 540, 561, 598, 599, 628, 640, 691, 716, 728, 753, 810, 820, 832, 834, 859, 883, 936, 948, 960, 972, 984, 996, 1008, 1020, 1211, 1223, 1259, 1271);
        
        // Convertir la lista a un array de enteros
        int[] nodeIndices = intersecciones.stream().mapToInt(Integer::intValue).toArray();

        while (!game.gameOver()) {
            if (tickLimit != -1 && tickLimit < game.getTotalTime()) {
                break;
            }
            handlePeek(game);
            game.advanceGame(
                    pacManController.getMove(getPacmanCopy(game), System.currentTimeMillis() + timeLimit),
                    ghostControllerCopy.getMove(getGhostsCopy(game), System.currentTimeMillis() + timeLimit));
            
            // Llamar a la función con el array
	        GameView.addPoints(game, Color.RED, nodeIndices);

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
        
        return game.getScore();
    }
	
	

	public void runGameGenerateDataSet(Controller<MOVE> pacManController, GhostController ghostController, int iter, String fileName, boolean DEBUG) {
		
		int delay = 0;	//El delay entre las ejecuciones es 0, porque queremos que se ejecute lo mas rapido posible
		
		long inicio = 0;
		long lineasIniciales = 0;
		
		if(DEBUG) {
			inicio = System.nanoTime();
			lineasIniciales = DataSetRecorder.contarLineas(fileName);
		}
				
		for(int i = 0;i<iter;i++) {
			
			Game game = setupGame();

			// Instancia de la clase que recopila los datos
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
				
				
				// RECOPILAR EL ESTADO DEL JUEGO
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
			

			// Guardo los datos
			try {
				dataRecorder.saveDataToCsv(fileName, true);
				
				if(DEBUG) {
					System.out.println(i + ". Estados correctamente guardados en: " + fileName + ".csv con score: " + game.getScore());
				}
			} catch (IOException e) {
				e.printStackTrace();
			}

			postcompute(pacManController, ghostController);
		}
		
		if(DEBUG) {
			
			long fin = System.nanoTime();  // Tiempo final
	        long duracion = fin - inicio;  // Duracion en ns

	        // Convertir nanosegundos a segundos
	        long segundosTotales = duracion / 1_000_000_000;
	        long horas = segundosTotales / 3600;
	        long minutos = (segundosTotales % 3600) / 60;
	        long segundos = segundosTotales % 60;
	        
	        
	        
	        long lineasFinales = DataSetRecorder.contarLineas(fileName);
	        long lineasCreadas = lineasFinales - lineasIniciales;

	        System.out.println("\n\n[DEBUG] Información de ejecución:\n");
	        System.out.printf("\tTiempo total: %d horas, %d minutos, %d segundos%n", horas, minutos, segundos);
	        System.out.println("\tLineas iniciales: " + lineasIniciales + ", Lineas creadas: " + lineasCreadas + ", Lineas finales: " + lineasFinales);
		}
	}
	
	
	
	public int runGameSocketConection(Controller<MOVE> pacManController, GhostController ghostController, int delay) {
		Game game = setupGame();
		
		//Se crea una instancia de la clase que se encarga de procesar el estado actual del juego
		GameStateFilter gameFilter = new GameStateFilter(game);
		
		
		// Instancia de la clase que maneja el socket
		
		String host = "localhost";
	    int port = 12345;
	    Socket socket;
	    PrintWriter out;
	    BufferedReader in;
	    try {
	        socket = new Socket(host, port);
	        out = new PrintWriter(socket.getOutputStream(), true);
	        in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
	    } catch (Exception e) {
	        System.out.println("Error al conectar con el servidor: " + e.getMessage());
	        return -1; // Termina si no se puede conectar
	    }
		
		
		precompute(pacManController, ghostController);

		GameView gv = (visuals) ? setupGameView(pacManController, game) : null;

		GhostController ghostControllerCopy = ghostController.copy(ghostPO);

		while (!game.gameOver()) {
			if (tickLimit != -1 && tickLimit < game.getTotalTime()) {
				break;
			}
			handlePeek(game);		
			
			
			MOVE pacmanMove = MOVE.NEUTRAL;
			
			
			// Solo pide calcular el movimineto de Pacman, cuando pasa por una interseccion
			if(game.isJunction(game.getPacmanCurrentNodeIndex())) {
				
				// Obtener gamaState filtrado
				String filteredGameState = gameFilter.getActualGameState();				
				
				try {
					// Pasar gameState filtrado por el socket				
					out.println(filteredGameState);

					// Obtener respuesta del socket con el movimiento
					String respuesta = in.readLine();
					
					System.out.println(respuesta);
					
					pacmanMove = MOVE.valueOf(respuesta);
					
				} catch (Exception e) {
		            System.out.println("Error durante la comunicación con el servidor: " + e.getMessage());
		            break; // Finaliza el bucle si ocurre un error de comunicación
		        }
			}							
			
			/*
			game.advanceGame(pacManController.getMove(getPacmanCopy(game), System.currentTimeMillis() + timeLimit),
					ghostControllerCopy.getMove(getGhostsCopy(game), System.currentTimeMillis() + timeLimit));
			*/
			
			// Pacman ejecuta ese movimiento
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
		System.out.println(game.getScore());

		postcompute(pacManController, ghostController);

		return game.getScore();
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