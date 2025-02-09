package pacman.game.socket;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;

import pacman.controllers.GhostController;
import pacman.controllers.PacmanController;

public class socketPython {
    private String host;
    private int port;
    private Socket socket;
    private PrintWriter out;
    private BufferedReader in;

    public socketPython(String host, int port) throws Exception {
        this.host = host;
        this.port = port;
        connect();
    }

    private void connect() throws Exception {
        try {
            socket = new Socket(host, port);
            out = new PrintWriter(socket.getOutputStream(), true);
            in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        } catch (Exception e) {
            throw new Exception("Error al conectar con el servidor: " + e.getMessage());
        }
    }

    public String sendGameState(String gameState) {
        try {
            out.println(gameState);
            return in.readLine(); // Recibir respuesta del modelo
        } catch (Exception e) {
            System.out.println("Error en la comunicación con el servidor: " + e.getMessage());
            return "NEUTRAL"; // Valor por defecto en caso de error
        }
    }

    public void close() {
        try {
            if (socket != null) socket.close();
            if (out != null) out.close();
            if (in != null) in.close();
        } catch (Exception e) {
            System.out.println("Error al cerrar el socket: " + e.getMessage());
        }
    }
}
