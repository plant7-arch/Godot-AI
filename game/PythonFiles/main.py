#Testing 
import time
t = 0

import os
import struct
import socket
from typing import Tuple
#FOR UDS CONNECTION

OK = 1
ERR_CLIENT_DISCONNECTED = 0

DONE = True

SERVER_ADDRESS = './uds_socket'

def recvall(client: socket.socket, size: int, chunk_size: int = 1024) -> Tuple[bool, bytearray]:

    buffer = bytearray()
    while size - len(buffer) > chunk_size:
        chunk = client.recv(chunk_size)
        if not chunk: 
            return (ERR_CLIENT_DISCONNECTED, buffer)
        buffer.extend(chunk)

    lastchunk = client.recv(size - len(buffer))
    if not lastchunk: 
        return (ERR_CLIENT_DISCONNECTED, buffer)
    buffer.extend(lastchunk)

    return (OK, buffer)

try:
    os.unlink(SERVER_ADDRESS)
except OSError:
    if os.path.exists(SERVER_ADDRESS):
        raise

with socket.socket(socket.AF, socket.SOCK_STREAM) as server:

    server.bind((IP, PORT))
    server.setblocking(True)
    server.listen(1)

    client, address = server.accept()
    client.setblocking(True)

    with client:
        (observation_size,) = struct.unpack('<I', client.recv(4))
        (format_size,) = struct.unpack('<I', client.recv(4))

        (status, format_buffer) = recvall(client, format_size)
        format_ = format_buffer.decode()

        while True:
            (status, buffer) = recvall(client, observation_size)
            if status != OK:
                break

            # Testing
            print(struct.unpack(format_, buffer))
            
            (screenshot_size,) = struct.unpack('<I', client.recv(4))
            (status, screenshot_buffer) = recvall(client, screenshot_size)
            if status != OK: 
                break

            # Testing
            print(f"{len(screenshot_buffer)=}")
            with open(f"pictures\\received_screenshot{t}.png", "wb") as img_file:
                img_file.write(screenshot_buffer)

            if t>0:
                (reward,) = struct.unpack('<i', client.recv(4))
                print("Reward: ", reward)
            
            print('Waiting...')
            time.sleep(1)
            t+=1



            client.sendall(struct.pack('?', DONE))











# FOR TCP CONNECTION

# OK = 1
# ERR_CLIENT_DISCONNECTED = 0

# DONE = True

# IP = "127.0.0.1"
# PORT = 6060

# def recvall(client: socket.socket, size: int, chunk_size: int = 1024) -> Tuple[bool, bytearray]:

#     buffer = bytearray()
#     while size - len(buffer) > chunk_size:
#         chunk = client.recv(chunk_size)
#         if not chunk: 
#             return (ERR_CLIENT_DISCONNECTED, buffer)
#         buffer.extend(chunk)

#     lastchunk = client.recv(size - len(buffer))
#     if not lastchunk: 
#         return (ERR_CLIENT_DISCONNECTED, buffer)
#     buffer.extend(lastchunk)

#     return (OK, buffer)


# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:

#     server.bind((IP, PORT))
#     server.setblocking(True)
#     server.listen(1)

#     client, address = server.accept()
#     client.setblocking(True)

#     with client:
#         (observation_size,) = struct.unpack('<I', client.recv(4))
#         (format_size,) = struct.unpack('<I', client.recv(4))

#         (status, format_buffer) = recvall(client, format_size)
#         format_ = format_buffer.decode()

#         while True:
#             (status, buffer) = recvall(client, observation_size)
#             if status != OK:
#                 break

#             # Testing
#             print(struct.unpack(format_, buffer))
            
#             (screenshot_size,) = struct.unpack('<I', client.recv(4))
#             (status, screenshot_buffer) = recvall(client, screenshot_size)
#             if status != OK: 
#                 break

#             # Testing
#             print(f"{len(screenshot_buffer)=}")
#             with open(f"pictures\\received_screenshot{t}.png", "wb") as img_file:
#                 img_file.write(screenshot_buffer)

#             if t>0:
#                 (reward,) = struct.unpack('<i', client.recv(4))
#                 print("Reward: ", reward)
            
#             print('Waiting...')
#             time.sleep(1)
#             t+=1



#             client.sendall(struct.pack('?', DONE))







