import socket, json, sys

def send(payload, timeout=20):
    s = socket.create_connection(('127.0.0.1', 9876), timeout=timeout)
    s.sendall(json.dumps(payload).encode('utf-8'))
    s.settimeout(timeout)
    data = b''
    try:
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    except socket.timeout:
        pass
    s.close()
    try:
        return json.loads(data.decode('utf-8', errors='replace'))
    except Exception:
        return data.decode('utf-8', errors='replace')

if __name__ == '__main__':
    msg_type = sys.argv[1]
    if msg_type == 'execute_code':
        code_file = sys.argv[2]
        with open(code_file, 'r', encoding='utf-8') as f:
            code = f.read()
        result = send({'type': 'execute_code', 'params': {'code': code}})
    else:
        result = send({'type': msg_type, 'params': {}})
    print(json.dumps(result, indent=2)[:6000])
