package pacman.game.consolePrinter;

import java.util.Scanner;

public class UserPrompt {

    private static final Scanner scanner = new Scanner(System.in);

    /**
     * Metodo para solicitar una confirmacion al usuario con un mensaje personalizado
     * @param mensaje Mensaje que se muestra en consola
     * @return `true` si el usuario responde "S", `false` si responde "N"
     */
    public static boolean confirmar(String mensaje) {
        String respuesta;
        
        while (true) {
            System.out.print(mensaje + " (S/N): ");
            respuesta = scanner.nextLine().trim().toUpperCase();

            if (respuesta.equals("S")) {
                return true;
            } else if (respuesta.equals("N")) {
                return false;
            } else {
                System.out.println("Por favor, ingresa 'S' para Si o 'N' para No");
            }
        }
    }
}
