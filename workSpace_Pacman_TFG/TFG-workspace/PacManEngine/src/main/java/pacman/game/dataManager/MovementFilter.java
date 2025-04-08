package pacman.game.dataManager;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import pacman.game.Constants.MOVE;
import pacman.game.Game;

public class MovementFilter {
	
    public static List<MOVE> getValidMoves(Game game) {
    	
        // Movimientos posibles
        MOVE[] possibleMoves = game.getPossibleMoves(game.getPacmanCurrentNodeIndex());
        List<MOVE> validMoves = new ArrayList<>(Arrays.asList(possibleMoves));
        
        // Quitar el movimiento para la vuelta atras
        MOVE lastMove = game.getPacmanLastMoveMade();
        MOVE oppositeMove = lastMove.opposite();
        validMoves.remove(oppositeMove);
        
        return validMoves;
    }
}
