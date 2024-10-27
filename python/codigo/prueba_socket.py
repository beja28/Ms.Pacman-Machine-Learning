import socket
import json

#Configuración
HOST = 'localhost'  # Podemos poner una IP o localhost
PORT = 12345        # Puerto que queramos usar

#Crear socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))

#Escuchar conexiones entrantes
server_socket.listen()

print(f"Servidor escuchando en {HOST}:{PORT}")

#Aceptar conexión
conn, addr = server_socket.accept()
print(f"Conectado con {addr}")

#Recibir datos
while True:
    data = conn.recv(1024)  # El 1024 es el número máximo de bytes que se intenta recibir
    if not data:
        break

    mensaje = data.decode('utf-8')
    print(f"Datos recibidos: {mensaje}")

    # Responder con algo diferente según lo recibido
    if mensaje:
        respuesta = f"NEUTRAL\n"  # Agrega un salto de línea al final
        conn.sendall(respuesta.encode('utf-8'))  # Enviar respuesta codificada en UTF-8

conn.close()