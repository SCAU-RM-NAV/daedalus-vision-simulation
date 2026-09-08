PORT=${PORT:-/dev/ttyUSB0} python3 - <<'PY'
import os, struct, serial
port = os.environ["PORT"]
ser = serial.Serial(port, 115200, timeout=0.2)
buf = bytearray()
FRAME_LEN = 44

def crc16(data):
    crc = 0xffff
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if (crc & 1) else (crc >> 1)
            crc &= 0xffff
    return crc

ok = bad = 0
print("reading", port)
while ok < 20:
    buf += ser.read(256)
    while len(buf) >= FRAME_LEN:
        i = buf.find(b"SP")
        if i < 0:
            buf.clear()
            break
        if i:
            del buf[:i]
        if len(buf) < FRAME_LEN:
            break
        frame = bytes(buf[:FRAME_LEN])
        recv_crc = frame[-2] | (frame[-1] << 8)
        calc_crc = crc16(frame[:-2])
        if calc_crc != recv_crc:
            bad += 1
            if bad <= 10 or bad % 100 == 0:
                raw = " ".join(f"{b:02X}" for b in frame)
                print(f"bad={bad} recv_crc=0x{recv_crc:04x} calc_crc=0x{calc_crc:04x} raw={raw}")
            del buf[0]
            continue
        del buf[:FRAME_LEN]
        vals = struct.unpack("<2sB4f5fHBH", frame)
        _, mode, qw, qx, qy, qz, yaw, yaw_vel, pitch, pitch_vel, bs, bc, camp, _ = vals
        ok += 1
        print(f"ok={ok:03d} bad={bad} mode={mode} camp={camp} yaw={yaw:.3f} pitch={pitch:.3f} yaw_vel={yaw_vel:.3f} pitch_vel={pitch_vel:.3f} bullet={bs:.1f} count={bc}")
PY
