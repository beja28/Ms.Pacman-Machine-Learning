package es.ucm.fdi.ici.TFGpacman;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;

import pacman.game.consolePrinter.MessagePrinter;

public class SocketPython {
    private String host;
    private int port;
    private Socket socket;
    private PrintWriter out;
    private BufferedReader in;
    private MessagePrinter printer;

    public SocketPython(String host, int portprivate, MessagePrinter ptr) throws Exception {
        this.host = host;
        this.port = portprivate;
        this.printer = ptr;
        connect();
    }

    private void connect() throws Exception {
        try {
        	if (socket != null && !socket.isClosed()) {
                close();  // Cerrar conexión previa si existe
            }
        	
            socket = new Socket(this.host, this.port);
            out = new PrintWriter(socket.getOutputStream(), true);
            in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            
            printer.mostrarInfo("Conectado a Python en " + host + ":" + port);
        } catch (Exception e) {
        	printer.mostrarError("Al conectar con el servidor");
        	throw new Exception(e.getMessage());
        }
    }
    

    public String sendAndReceivePrediction(String gameState) {
        try {
        	
        	if (socket == null || socket.isClosed()) {
        		printer.mostrarAdvertencia("Conexión perdida. Intentando reconectar...");
                connect();
            }
        	
            out.println(gameState);	//Enviar
            out.flush();	//Se asegura que el mensaje se envia de forma inmediata
            
            String response =  in.readLine(); // Recibir respuesta del modelo
            
            if (response == null) {
            	printer.mostrarError("En la comunicación con el servidor (respuesta del servidor nula) ==> Siguiente movimiento <NEUTRAL>'");
                return "NEUTRAL"; // Valor por defecto si no hay respuesta
            }
            
            return response;
            
        } catch (Exception e) {
        	printer.mostrarError("En la comunicación con el servidor ==> Siguiente movimiento <NEUTRAL>'");
            System.out.println(e.getMessage());
            return "NEUTRAL"; // Valor por defecto en caso de error
        }
    }
    

    public void close() {
        try {
            if (socket != null) socket.close();
            if (out != null) out.close();
            if (in != null) in.close();
            printer.mostrarInfo("Conexión cerrada");
        } catch (Exception e) {
        	printer.mostrarError("Al cerrar el socket");
            System.out.println(e.getMessage());
        }
    }
}
