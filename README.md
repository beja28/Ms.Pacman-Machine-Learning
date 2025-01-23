# Ms.Pacman-Machine-Learning

## Requisitos del Proyecto

Este repositorio incluye un archivo `requirements.txt` que especifica las versiones necesarias de las librerías utilizadas. Esto permite crear un entorno virtual consistente para trabajar con el proyecto.

### Cómo configurar el entorno

1. **Crear un entorno virtual**:
   Abre una terminal en la raíz del proyecto y ejecuta:

   ```bash
   python -m venv .venv
   ```

   También puedes abrir la paleta de comandos en VSCode (Ctrl + Shift + P)
   y buscar "Python: Create Terminal" para abrir una terminal con el entorno virtual activado.

   Seleccionar la versión de Python correspondiente al entorno virtual creado, y luego seleccionar el archivo requirements.txt para instalar las dependencias.

2. **Activar el entorno virtual**:

   - En Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - En macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

3. **Instalar las dependencias**:
   Con el entorno virtual activado, instala las librerías necesarias:
   ```bash
   pip install -r requirements.txt
   ```

### Notas

- Asegúrate de tener una versión de Python compatible (3.8 o superior se recomienda).
- Si encuentras problemas al activar el entorno virtual en Windows, verifica que la ejecución de scripts está habilitada:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```
  Esto permite ejecutar scripts locales.
