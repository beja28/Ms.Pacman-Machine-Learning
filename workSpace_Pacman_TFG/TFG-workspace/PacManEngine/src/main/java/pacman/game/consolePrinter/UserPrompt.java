package pacman.game.consolePrinter;

import java.util.Scanner;

public class UserPrompt {

    private static final Scanner scanner = new Scanner(System.in);
    private static MessagePrinter printer;

    public UserPrompt(MessagePrinter printer) {
        UserPrompt.printer = printer;
    }

    public static boolean confirmar(String mensaje) {
        String respuesta;

        while (true) {
            System.out.print(mensaje + " (S/N): ");
            respuesta = scanner.nextLine();

            if (respuesta == null || respuesta.trim().isEmpty()) {
            	printer.mostrarAdvertencia("Entrada no válida. Por favor, ingresa 'S' para Sí o 'N' para No.");
                continue;
            }

            respuesta = respuesta.trim().toUpperCase();

            if (respuesta.equals("S")) {
            	System.out.println();
                return true;
            } else if (respuesta.equals("N")) {
            	System.out.println();
                return false;
            } else {
            	printer.mostrarAdvertencia("Entrada no válida. Por favor, ingresa 'S' para Sí o 'N' para No.");
            }
        }
    }

    public static boolean solicitarConfirmacionEjecucion() {
        boolean continuar = confirmar("¿Deseas continuar con la ejecución?");
        if (!continuar) {
        	printer.mostrarInfo("Ejecución cancelada por el usuario.");
        }
        return continuar;
    }
}
