import struct

byte_stream = struct.pack('>2i', 1771, 3)
print(byte_stream)
print(struct.calcsize('f'))
# print(int.from_bytes(byte_stream, 'big'))
print(struct.unpack('>2i', byte_stream))

# hi()
# def hi():print('HI')


# Python program creating a
# context manager
    
class ContextManager():
    def __init__(self):
        print('init method called')
            
    def __enter__(self):
        print('enter method called')
        return self
        
    def __exit__(self, exc_type, exc_value, exc_traceback):
        print('exit method called')
    
    
with ContextManager() as manager:
    manager.__exit__(1, 2, 3)
    print('with statement block')