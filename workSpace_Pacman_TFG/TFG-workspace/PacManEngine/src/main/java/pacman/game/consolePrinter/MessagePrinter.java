package pacman.game.consolePrinter;


public class MessagePrinter {
    private boolean debugMode;

    // Códigos ANSI para colores y formato de texto
    private static final String RESET = "\u001B[0m";
    private static final String BOLD = "\u001B[1m";
    private static final String UNDERLINE = "\u001B[4m";
    private static final String BLUE = "\u001B[34m";
    private static final String YELLOW = "\u001B[33m";
    private static final String RED = "\u001B[31m";
    private static final String GREEN = "\u001B[32m";
    private static final String CYAN = "\u001B[36m";
    private static final String MAGENTA = "\u001B[35m";

    /**
     * Constructor para inicializar el MessagePrinter con modo de depuración activado o desactivado.
     * @param debugMode Si es true, se mostrarán los mensajes de depuración; si es false, no.
     */
    public MessagePrinter(boolean debugMode) {
        this.debugMode = debugMode;
    }

    /**
     * Muestra un mensaje informativo en consola con formato azul y negrita.
     * @param mensaje Mensaje a mostrar.
     */
    public void mostrarInfo(String mensaje) {
        System.out.println(BOLD + BLUE + "[INFO] " + mensaje + RESET);
    }

    /**
     * Muestra un mensaje de advertencia en consola con color amarillo y negrita.
     * @param mensaje Mensaje a mostrar.
     */
    public void mostrarAdvertencia(String mensaje) {
        System.out.println(BOLD + YELLOW + "[ADVERTENCIA] " + mensaje + RESET);
    }

    /**
     * Muestra un mensaje de error en consola con color rojo y negrita.
     * @param mensaje Mensaje a mostrar.
     */
    public void mostrarError(String mensaje) {
        System.out.println(BOLD + RED + "[ERROR] " + mensaje + RESET);
    }

    /**
     * Muestra un mensaje de éxito en consola con color verde y negrita.
     * @param mensaje Mensaje a mostrar.
     */
    public void mostrarExito(String mensaje) {
        System.out.println(BOLD + GREEN + "[ÉXITO] " + mensaje + RESET);
    }

    /**
     * Muestra un mensaje de depuración en consola con color magenta y subrayado,
     * pero solo si el modo de depuración está activado.
     * @param mensaje Mensaje a mostrar.
     */
    public void mostrarDebug(String mensaje) {
        if (debugMode) {
            System.out.println(UNDERLINE + MAGENTA + "[DEBUG] " + mensaje + RESET);
        }
    }

    /**
     * Muestra un mensaje personalizado con tabulación.
     * @param mensaje Mensaje a mostrar.
     * @param nivelTabulacion Número de tabulaciones antes del mensaje.
     */
    public void mostrarConTabulacion(String mensaje, int nivelTabulacion) {
        String tabulaciones = "\t".repeat(Math.max(0, nivelTabulacion));
        System.out.println(tabulaciones + mensaje);
    }

    /**
     * Muestra un mensaje resaltado con color cyan y en negrita.
     * @param mensaje Mensaje a mostrar.
     */
    public void mostrarResaltado(String mensaje) {
        System.out.println(BOLD + CYAN + mensaje + RESET);
    }

    /**
     * Permite activar o desactivar el modo de depuración en tiempo de ejecución.
     * @param debugMode Nuevo valor para el modo de depuración.
     */
    public void setDebugMode(boolean debugMode) {
        this.debugMode = debugMode;
    }
}

